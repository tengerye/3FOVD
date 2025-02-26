import argparse
import glob
import pickle
import math
from collections import defaultdict
import supervision as sv
import numpy as np
import torch
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, annotate
import cv2
import json
import pandas as pd


def read_json(file_name):
    #Read JSON file
    with open(file_name) as infile:
        data = json.load(infile)
    return data


def sort_boxes_by_score(boxes, labels, scores, max_elem):
    sorted_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    sorted_boxes = [boxes[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    sorted_dict = {
        'boxes': sorted_boxes[:],  # :max_elem
        'labels': sorted_labels[:],
        'scores': sorted_scores[:],
    }

    return sorted_dict


from typing import List


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


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def preprocess_caption_car(caption: str) -> str:
    caption = "car" + "#" + caption
    return preprocess_caption(caption)


# IMP2 NMS
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



def draw_grounding_output(tokenizer, caption, boxes, logits_filt, image_source, output_path, token_threshold=0):
    # 截取目标词汇对应的logits索引
    # 初始化分词器
    tokenized = tokenizer(caption)
    input_ids = tokenized["input_ids"]
    phrases = []
    res_logits, token_index = logits_filt.max(dim=1)
    if token_threshold == 0:
        # 获取每一个bounding box对应的分值最高的token (一个bbox对应一个token)
        for index, box in zip(token_index, boxes):
            token_id = input_ids[index]
            phrases.append(tokenizer.decode(token_id))
    else:
        # 获取每一个bounding box对应的分值大于threshold的所有token
        for logit, box in zip(logits_filt, boxes):
            map = logit > token_threshold
            map[0: 1] = False
            map[255:] = False
            non_zero_idx = map.nonzero(as_tuple=True)[0].tolist()
            token_ids = [input_ids[i] for i in non_zero_idx]
            phrases.append(tokenizer.decode(token_ids))

    frame = annotate(np.asarray(image_source), boxes.numpy(), res_logits.numpy(), phrases)
    cv2.imwrite(output_path, frame)


# baseline
def get_grounding_output_baseline(model, image, data_type, captions, image_source, label_ls, box_threshold=0):
    device = "cuda"
    image = image.to(device)

    if data_type == "car":
        captions = [preprocess_caption_car(caption) for caption in captions]
    else:
        captions = [preprocess_caption(caption) for caption in captions]
    images = image[None].repeat(len(captions), 1, 1, 1)

    with torch.no_grad():
        outputs = model(images, captions=captions)

    logits_batched = [x.cpu().sigmoid() for x in outputs['pred_logits']]
    boxes_batched = [x.cpu() for x in outputs['pred_boxes']]

    res_pred_boxes = []
    res_pred_scores = []
    res_pred_labels = []
    for logits, boxes, label_id, caption in zip(logits_batched, boxes_batched, label_ls, captions):

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 1
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        pred_boxes = []
        for box in boxes_filt:
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor(
                [image_source.size[0], image_source.size[1], image_source.size[0], image_source.size[1]])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            pred_boxes.append(box)
        if len(pred_boxes) == 0:
            continue
        tmp_boxes = torch.stack(pred_boxes, dim=0)

        # draw_grounding_output(model.tokenizer, caption, tmp_boxes, logits_filt, image_source,
        #                                         f"./image/exp2/test_{label_id}_baseline.jpg")

        for box, score in zip(tmp_boxes, logits_filt):
            res_pred_boxes.append(box)
            res_pred_scores.append(score.max().item())
            res_pred_labels.append(label_id)

    return sort_boxes_by_score(res_pred_boxes, res_pred_labels, res_pred_scores, 1e6)


# IMP2
def get_grounding_output_IMP2(model, image, data_type, captions, image_source, label_ls, box_threshold=0):
    device = "cuda"
    image = image.to(device)
    # caption 预处理   preprocess_caption_car
    if data_type == "car":
        captions = [preprocess_caption_car(caption) for caption in captions]
    else:
        captions = [preprocess_caption(caption) for caption in captions]
    images = image[None].repeat(len(captions), 1, 1, 1)
    # 输入模型获取预测结果
    with torch.no_grad():
        outputs = model(images, captions=captions)
    logits_batched = [x.cpu().sigmoid() for x in outputs['pred_logits']]
    boxes_batched = [x.cpu() for x in outputs['pred_boxes']]

    res_pred_boxes = []
    res_pred_labels = []
    res_pred_scores = []
    # 遍历每一个caption的结果
    for logits, boxes, label_id, caption in zip(logits_batched, boxes_batched, label_ls, captions):

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold  # 将预测置信度小于box_threshold的行进行过滤
        logits_filt_256 = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        # 获取最大值对应的索引位置(即对应类别)以及分值(全局最高分值)
        logits_filt, max_index = logits_filt_256.max(dim=1)

        pred_boxes = []
        # 对bbox进行格式转换
        for box in boxes_filt:
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([image_source.size[0], image_source.size[1], image_source.size[0], image_source.size[1]])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            pred_boxes.append(box)
        if len(pred_boxes) == 0:
            continue
        tmp_boxes = torch.stack(pred_boxes, dim=0)

        res_index = nms_self(tmp_boxes, logits_filt, 0.5, max_index)
        tmp_boxes, logits_filt_res = (torch.index_select(tmp_boxes, dim=0, index=res_index),
                                  torch.index_select(logits_filt_256, dim=0, index=res_index))
        # draw_grounding_output(model.tokenizer, caption, tmp_boxes, logits_filt_res, image_source,
        #                       f"./image/exp2/vehicle/test_{label_id}_imp2.jpg")
        for box, score in zip(tmp_boxes, logits_filt_res):
            res_pred_boxes.append(box)
            res_pred_scores.append(score.max().item())
            res_pred_labels.append(label_id)

    return sort_boxes_by_score(res_pred_boxes, res_pred_labels, res_pred_scores, 1e6)


def saveObject(obj, path):
    print("Saving " + path + '.pkl')
    with open(path + ".pkl", 'wb') as fid:
        pickle.dump(obj, fid)


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


def get_caption(csv_path):
    cat2caption = {}
    data = pd.read_csv(csv_path)
    for _, row in data.iterrows():
        cat_name = row["Category_Name"]
        caption = row["Category_Caption(English)"]
        cat2caption[cat_name] = caption
    return cat2caption


def gen_predict_data(img_dir, annotation_filepath, BOX_TRESHOLD, type, output_path, data_type, csv_path):
    # 模型加载
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    model = model.to("cuda")
    img_ls = glob.glob(f"{img_dir}*.jpg")
    data = read_json(annotation_filepath)
    cat2caption = get_caption(csv_path)
    img2cat_dict = getImage2CatsMap(data)
    cats_dict = getCatsId2Item(data)
    complete_img_output = []
    img_processed = 0
    batch_size = 10
    for img_path in tqdm(img_ls[:]):
        image_source, image = load_image(img_path)
        filename = img_path.split("/")[-1]
        rel_cat_ids = img2cat_dict[filename]
        caption_ls, label_ls = [], []
        for id in rel_cat_ids:
            caption_ls.append(cats_dict[id]["name"] + " " + cat2caption[cats_dict[id]["name"]])
            label_ls.append(cats_dict[id]["id"])
        loop = math.ceil(len(caption_ls) / batch_size)
        img_output = {
            "boxes": [],
            "labels": [],
            "scores": [],
            "img_path": img_path
        }
        for i in range(loop):
            batch_caption_list = caption_ls[i * batch_size: (i + 1) * batch_size]
            batch_label_list = label_ls[i * batch_size: (i + 1) * batch_size]
            res = None
            if type == "baseline":
                res = get_grounding_output_baseline(model, image, data_type, batch_caption_list, image_source,
                                                    box_threshold=BOX_TRESHOLD, label_ls=batch_label_list)
            elif type == "imp2":
                res = get_grounding_output_IMP2(model, image, data_type, batch_caption_list, image_source,
                                                box_threshold=BOX_TRESHOLD, label_ls=batch_label_list)
            if res is None:
                raise Exception("unknown exp type")
            img_output["boxes"].extend(res["boxes"])
            img_output["labels"].extend(res["labels"])
            img_output["scores"].extend(res["scores"])

        img_processed += 1
        complete_img_output.append(img_output)

    output_path_file = f"{output_path}/dino_{type}_{data_type}_result"
    saveObject(complete_img_output,
               output_path_file)
    print(f"save success，the save path is: {output_path_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, help='图片路径')
    parser.add_argument('--anno_path', type=str, required=True, help='标注文件路径')
    parser.add_argument('--csv_path', type=str, required=True, help='提示词文件存储路径')
    parser.add_argument('--exp_type', type=str, required=True, help='实验类型，基准还是IMP2')
    parser.add_argument('--out', type=str, required=True, help='结果保存路径文件夹')
    parser.add_argument('--data_type', type=str, required=True, help='数据集类型，车辆数据集传car，商品数据集传product')
    args = parser.parse_args()
    img_dir, annotation_filepath, exp_type, output_path, data_type = args.img_dir, args.anno_path, args.exp_type, args.out, args.data_type
    csv_path = args.csv_path
    BOX_TRESHOLD = 0.2
    gen_predict_data(img_dir, annotation_filepath, BOX_TRESHOLD, exp_type, output_path, data_type, csv_path)

if __name__ == '__main__':
    main()
