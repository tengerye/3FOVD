import argparse
import glob
import re
from collections import defaultdict
from typing import List
import pandas as pd
import torch
from tqdm import tqdm
# Setup detectron2 logger
import supervision as sv
import sys
import numpy as np
import json, cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

MAX_DETECTION_PER_CATEGORY = 100
SCORE_THRESH = 0.50


def saveObject(obj, path):
    """"Save an object using the pickle library on a file

    :param obj: undefined. Object to save
    :param fileName: str. Name of the file of the object to save
    """
    print("Saving " + path + '.pkl')
    with open(path + ".pkl", 'wb') as fid:
        pickle.dump(obj, fid)

def read_json(file_name):
    #Read JSON file
    with open(file_name) as infile:
        data = json.load(infile)
    return data

def create_detector(config_path, weight_path):
    # Build the detector and download our pretrained weights
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH  # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False  # For better visualization purpose. Set to False for all classes.
    cfg.TEST.DETECTIONS_PER_IMAGE = 256
    predictor = DefaultPredictor(cfg)
    return predictor


def get_clip_embeddings(text_encoder, vocabulary, prompt=''):
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


def evaluate_image(predictor, text_encoder, im, vocabulary):
    classifier = get_clip_embeddings(text_encoder, vocabulary)
    num_classes = len(vocabulary)
    reset_cls_test(predictor.model, classifier, num_classes)
    # Run model and show results
    outputs = predictor(im)
    return outputs


def get_category_name(id, categories):
    for category in categories:
        if id == category['id']:
            return category['name']


def get_image_filepath(id, images):
    for image in images:
        if id == image['id']:
            return image['file_name']


def create_vocabulary(ann, categories):
    vocabulary_id = [ann['category_id']] + ann['neg_category_ids']
    vocabulary = [get_category_name(id, categories) for id in vocabulary_id]

    return vocabulary, vocabulary_id


def adjust_out_id(output, vocabulary_id):
    for i in range(len(output['instances'].pred_classes)):
        output['instances'].pred_classes[i] = vocabulary_id[output['instances'].pred_classes[i]]
    return output


def convert_to_standard_format(output):
    return {
        'labels': output['instances'].pred_classes.cpu().numpy().tolist(),
        'boxes': output['instances'].pred_boxes.tensor.cpu().numpy().tolist(),
        'scores': output['instances'].scores.cpu().numpy().tolist(),
        'total_scores': output['instances'].total_scores.cpu().numpy().tolist(),
        'category_id': output['category_id'],
        'image_filepath': output['image_filepath']
    }


def convert_to_standard_format_complete(outputs, img_path, max_elem=3000):
    img_std_out = {
        "labels": [],
        "boxes": [],
        "scores": [],
        "image_filepath": img_path
    }
    scores = []
    labels = []
    boxes = []
    for output in outputs:
        labels.extend(output['instances'].pred_classes.cpu().numpy().tolist())
        boxes.extend(output['instances'].pred_boxes.tensor.cpu().numpy().tolist())
        scores.extend(output['instances'].scores.cpu().numpy().tolist())

    # 按照分数从高到底进行截断
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_elem]
    for index in indices:
        img_std_out["labels"].append(labels[index])
        img_std_out["scores"].append(scores[index])
        img_std_out["boxes"].append(boxes[index])

    return img_std_out

def write_in_file(img_path):
    with open("./img_path.log", "a+") as f:
        f.write(img_path)

# img filename和类别对应关系的map
def getImage2CatsMap(annos):
    image_id_map = {}
    for image in annos["images"]:
        image_id_map[image["id"]] = image["file_name"]
    gt_dict = defaultdict(list)
    for item in annos['annotations']:
        gt_dict[image_id_map[item["image_id"]]].append(item["category_id"])
    return gt_dict


def getCatsId2Item(annos):
    res = {}
    for item in annos['categories']:
        res[item["id"]] = item
    return res

def preprocess_caption(caption, data_type):
    # 将caption分成word列表
    words = re.findall(r'\b\w+\b', caption.lower())
    words = list(set(words))
    if "car" not in words and data_type == "car":
        words.append("car")
    return words


def post_process_IMP1(output, vocabulary):
    target_words = ["car"]
    scores = output['instances']._fields["scores"]
    pred_classes = output['instances']._fields["pred_classes"]
    for idx, cls in enumerate(pred_classes):
        if vocabulary[cls].lower() in target_words and scores[idx] >= 0.8:
            scores[idx] = 1
    output['instances']._fields["scores"] = scores
    return output


def nms_self(bboxes, scores, iou_thresh, max_score_index_list):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)
    # 结果列表
    result = []
    index = torch.flip(scores.argsort(), [0])  # 对检测框按照置信度进行从高到低的排序，并获取索引
    # 下面的操作为了安全，都是对索引处理
    while len(index) > 0:
        # 当检测框不为空一直循环
        i = index[0]
        # 当前置信度最高的检测框对应的类别
        main_class = max_score_index_list[i]
        result.append(i)  # 将置信度最高的加入结果列表
        # 计算其他边界框与该边界框的IOU
        # 1. 计算当前框 与 其他框的交集面积
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11)
        h = np.maximum(0, y22 - y11)
        overlaps = w * h
        # 2. 根据 “交集面积 / 并集面积" 计算出iou值
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # 只保留满足IOU阈值的索引
        idx = np.where(ious <= iou_thresh)[0]
        # 将满足iou，且位置大小完全被当前置信度最高的框包含的小框过滤
        cover_index = []
        for k in range(len(idx)):
            no = idx[k] + 1
            original_index = index[no]
            if x1[original_index] > x1[i] and y1[original_index] > y1[i] and x2[original_index] < x2[i] and y2[
                original_index] < y2[i]:
                # 将完全重合的元素进行删除
                cover_index.append(k)
        idx = np.delete(idx, cover_index)
        # 将不满足IOU阈值的索引合并 （不同类别的候选框进行合并）
        merge_bbox_ids = []
        for j in range(len(ious)):
            cur_class_index = index[j + 1]
            cur_class = max_score_index_list[cur_class_index]
            # 将与当前框，iou超过threshold+不同类别的+不是包含关系的框进行合并
            if (j not in idx) and (cur_class != main_class) and (j not in cover_index):
                merge_bbox_ids.append(j)
        # 计算合并的区域
        x1_l, x2_l, y1_l, y2_l = [x1[i].item()], [x2[i].item()], [y1[i].item()], [y2[i].item()]
        for i2 in merge_bbox_ids:
            index2 = index[i2 + 1]
            x1_l.append(x1[index2].item())
            x2_l.append(x2[index2].item())
            y1_l.append(y1[index2].item())
            y2_l.append(y2[index2].item())
        bboxes[i, 0] = min(x1_l)
        bboxes[i, 1] = min(y1_l)
        bboxes[i, 2] = max(x2_l)
        bboxes[i, 3] = max(y2_l)
        index = index[idx + 1]  # 处理剩余的边框

    return torch.Tensor(result).long()


def post_process_IMP2(output, vocabulary):
    scores = output['instances']._fields["scores"].cpu()
    boxes = output['instances']._fields["pred_boxes"].tensor.cpu()
    pred_classes = output['instances']._fields["pred_classes"].cpu()
    class_list = [vocabulary[i] for i in pred_classes]
    indexes = nms_self(boxes, scores, 0.5, class_list)
    processed_boxes, processed_scores, processed_cls = (torch.index_select(boxes, dim=0, index=indexes),
                              torch.index_select(scores, dim=0, index=indexes),
                                            torch.index_select(pred_classes, dim=0, index=indexes))
    output['instances']._fields["scores"] = processed_scores
    output['instances']._fields["pred_classes"] = processed_cls
    output['instances']._fields["pred_boxes"].tensor = processed_boxes
    return output

def annotate(image_source: np.ndarray, boxes: np.ndarray, logits: np.ndarray, phrases: List[str]) -> np.ndarray:
    detections = sv.Detections(xyxy=boxes)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

def draw_grounding_output(vocabulary, vindex, boxes, logits_filt, image_source, output_path):
    phrases = [vocabulary[i] for i in vindex]
    boxes = np.array([i.tolist() for i in boxes])
    frame = annotate(np.asarray(image_source), boxes, logits_filt.cpu().numpy(), phrases)
    cv2.imwrite(output_path, frame)

def get_caption(csv_path):
    cat2caption = {}
    data = pd.read_csv(csv_path)
    for _, row in data.iterrows():
        cat_name = row["Category_Name"]
        caption = row["Category_Caption(English)"]
        cat2caption[cat_name] = caption
    return cat2caption

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='caption file')
    parser.add_argument('--img_path', type=str, required=True, help='Dataset to process')
    parser.add_argument('--anno_path', type=str, required=True, help='Dataset to process')
    parser.add_argument('--exp_type', type=str, required=True, help='Dataset to process')
    parser.add_argument('--data_type', type=str, required=True, help='Dataset to process')
    parser.add_argument('--out', type=str, default="detic", help='Out path')
    parser.add_argument('--config_path', type=str,
                        default="./configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
                        help='Path of the configuration file')
    parser.add_argument('--weight_path', type=str,
                        default="./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
                        help='Path of the weight file')
    args = parser.parse_args()

    exp_type = args.exp_type
    data_type = args.data_type
    csv_path = args.csv_path
    cat2caption = get_caption(csv_path)
    img_paths = args.img_path
    img_ls = glob.glob(f"{img_paths}*.jpg")
    data = read_json(args.anno_path)
    img2cat_dict = getImage2CatsMap(data)
    cats_dict = getCatsId2Item(data)

    predictor = create_detector(args.config_path, args.weight_path)
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()

    complete_outputs = []
    complete_outputs_gpu = []
    for img_path in tqdm(img_ls[:]):

        filename = img_path.split("/")[-1]
        rel_cat_ids = img2cat_dict[filename]
        image = cv2.imread(img_path)
        caption_ls, label_ls = [], []
        for id in rel_cat_ids:
            caption_ls.append(cats_dict[id]["name"].strip() + " " + cat2caption[cats_dict[id]["name"].strip()])
            label_ls.append(cats_dict[id]["id"])
        for i in range(len(caption_ls)):
            vocabulary = caption_ls[i]
            vocabulary_ids = label_ls[i]
            vocabulary = preprocess_caption(vocabulary, data_type)
            output = evaluate_image(predictor, text_encoder, image, vocabulary)
            if exp_type == "imp2":
                # IMP2
                output = post_process_IMP2(output, vocabulary)
            # 结果可视化
            # draw_grounding_output(vocabulary, output['instances']._fields['pred_classes'], output['instances']._fields['pred_boxes'], output['instances']._fields['scores'], image,
            #                       f"/Detic/image/vehicle/imp2/test_{filename.split('.')[0]}_{vocabulary_ids}_baseline.jpg")
            # 对输出调整label_id
            for i in range(len(output['instances']._fields['pred_classes'])):
                output['instances']._fields['pred_classes'][i] = vocabulary_ids
            complete_outputs_gpu.append(output)

        complete_outputs.append(convert_to_standard_format_complete(complete_outputs_gpu, img_path))
        complete_outputs_gpu = []
    output_path = args.out
    save_file = f"{output_path}/detic_{data_type}_{exp_type}_result"
    saveObject(complete_outputs, save_file)


if __name__ == '__main__':
    main()
