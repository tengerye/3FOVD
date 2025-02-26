import glob
import json
import re
import sys
from collections import defaultdict
from typing import List
import supervision as sv
import datasets.transforms as T
import argparse
import random
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
import torch
from models import build_model
from main import get_args_parser
from torchvision.ops import batched_nms
import pickle
import pandas as pd


def saveObject(obj, path):
    print("Saving " + path + '.pkl')
    with open(path + ".pkl", 'wb') as fid:
        pickle.dump(obj, fid)


def loadObject(path):
    try:
        with open(path + '.pkl', 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except IOError:
        return None

def read_json(file_name):
    # Read JSON file
    with open(file_name) as infile:
        data = json.load(infile)
    return data

device = 'cuda'

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
    for i in range(len(output['labels'])):
        output['labels'][i] = vocabulary_id[output['labels'][i]]
    return output


def get_image(img_path):
    """
    Read and normalize an image
    """
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]
    imm = Image.open(img_path)
    imm = ImageOps.exif_transpose(imm)
    image_data = imm
    # Get the original width and height
    w, h = imm.size
    normalize = T.Compose([
        T.ToRGB(),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])
    compose = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])
    imm, _ = compose(imm, None)
    imm = imm.unsqueeze(0)  # add batch dimension
    return image_data, imm, w, h


def apply_NMS(preds, iou=0.6):
    boxes = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']

    indexes_to_keep = batched_nms(torch.FloatTensor(boxes),
                                  torch.FloatTensor(scores),
                                  torch.IntTensor([0] * len(boxes)),
                                  iou)

    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []

    for x in range(len(boxes)):
        if x in indexes_to_keep:
            filtered_boxes.append(boxes[x])
            filtered_scores.append(scores[x])
            filtered_labels.append(labels[x])

    preds['boxes'] = filtered_boxes
    preds['scores'] = filtered_scores
    preds['labels'] = filtered_labels
    return preds

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
    scores = output["scores"].cpu()
    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    class_list = [vocabulary[i] for i in labels]
    indexes = nms_self(boxes, scores, 0.5, class_list)
    processed_boxes, processed_scores, processed_labels = (torch.index_select(boxes, dim=0, index=indexes),
                              torch.index_select(scores, dim=0, index=indexes),
                                            torch.index_select(labels, dim=0, index=indexes))
    output["scores"] = processed_scores
    output["labels"] = processed_labels
    output["boxes"] = processed_boxes
    return output

def convert_format(boxes, dim=2):
    if dim == 2:
        for box in boxes:
            box[2] += box[0]
            box[3] += box[1]
    else:
        boxes[2] += boxes[0]
        boxes[3] += boxes[1]
    return boxes


def evaluate_image(model, imm, post_processors, vocabulary, size, target_label, exp_flag="baseline", box_threshold=0.1, source_image=None):
    w, h = size
    target_sizes = torch.tensor([[h, w]], device=device)

    outputs = model(imm, categories=vocabulary)
    results = post_processors['bbox'](outputs, target_sizes)[0]
    if exp_flag == "imp2":
        results = post_process_IMP2(results, vocabulary)
    scores = results['scores'].tolist()
    labels = results['labels'].tolist()
    boxes = results['boxes'].tolist()
    preds = {
        'scores': [],
        'labels': [],
        'boxes': [],
    }

    for i, score in enumerate(scores):
        if score >= box_threshold:
            preds["scores"].append(scores[i])
            preds["labels"].append(labels[i])
            preds["boxes"].append(boxes[i])

    # draw_grounding_output(vocabulary, preds["labels"], preds["boxes"], preds["scores"], source_image,
    #                       f"/data/chaihaojiang/our_data/CORA/image/imp2/test_demo_{target_label}_imp2.jpg")

    # 将label对齐
    for i in range(len(preds["labels"])):
        preds["labels"][i] = target_label
    return preds

def filter_img_output(img_output, max_elem=3000):
    res_img_output = {
        "boxes": [],
        "labels": [],
        "scores": [],
        "img_path": img_output["img_path"]
    }
    scores = img_output["scores"]
    # 按照分数从高到底进行截断
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_elem]
    for index in indices:
        res_img_output["labels"].append(img_output["labels"][index])
        res_img_output["scores"].append(img_output["scores"][index])
        res_img_output["boxes"].append(img_output["boxes"][index])
    return res_img_output

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
    if "car" not in words and data_type == 'car':
        words.append("car")
    return words

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
    boxes = np.array(boxes)
    frame = annotate(np.asarray(image_source), boxes, np.array(logits_filt), phrases)
    cv2.imwrite(output_path, frame)


def get_caption(csv_path):
    cat2caption = {}
    data = pd.read_csv(csv_path)
    for _, row in data.iterrows():
        cat_name = row["Category_Name"]
        caption = row["Category_Caption(English)"]
        cat2caption[cat_name] = caption
    return cat2caption

def main(args):
    data = read_json(args.anno_path)

    # fix the seed for reproducibility
    # 给随机函数设置种子
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    csv_path = args.csv_path
    exp_type = args.exp_type
    data_type = args.data_type
    img_path = args.img_path
    # device = 'cuda'
    device = 'cuda'
    # 模型构建
    model, criterion, post_processors = build_model(args)
    model.to(device)
    model.eval()
    # 从检查点恢复继续执行
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_ema'])  # checkpoint['model_ema']
        print("checkpoint loaded")
    img2cat_dict = getImage2CatsMap(data)
    cats_dict = getCatsId2Item(data)
    cat2caption = get_caption(csv_path)
    complete_outputs = []
    # 数据遍历
    img_file_list = glob.glob(f"{img_path}*.jpg")
    img_processed = 0
    for img_path in tqdm(img_file_list[:1000]):
        image_data, imm, w, h = get_image(img_path)
        imm = imm.to(device)
        img_output = {
            "boxes": [],
            "labels": [],
            "scores": [],
            "img_path": img_path
        }
        filename = img_path.split("/")[-1]
        rel_cat_ids = img2cat_dict[filename]
        caption_ls, label_ls = [], []
        for id in rel_cat_ids:
            caption_ls.append(cats_dict[id]["name"] + " " + cat2caption[cats_dict[id]["name"]])
            label_ls.append(cats_dict[id]["id"])
        for i in range(len(caption_ls)):
            caption = caption_ls[i]
            label = label_ls[i]
            vocabulary = preprocess_caption(caption, data_type)
            output = evaluate_image(model, imm, post_processors, vocabulary, (w, h), label, exp_flag=exp_type, source_image=image_data)
            img_output["boxes"].extend(output["boxes"])
            img_output["labels"].extend(output["labels"])
            img_output["scores"].extend(output["scores"])
        complete_outputs.append(img_output)
        img_processed += 1

    output_file_path = f"{args.out}/cora_{exp_type}_{data_type}_result"
    saveObject(complete_outputs, output_file_path)


def start():
    # python inference_on_benchmark.py --backbone clip_RN50
    parser = argparse.ArgumentParser("CORA", parents=[get_args_parser()])
    parser.add_argument('--anno_path', type=str, required=True, help='Dataset to process')
    parser.add_argument('--out', type=str, required=True, help='Out path')
    parser.add_argument('--csv_path', type=str, required=True, help='caption csv_path')
    parser.add_argument('--data_type', type=str, required=True)
    parser.add_argument('--exp_type', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)
    args = parser.parse_args()

    main(args)

if __name__ == '__main__':
    start()



