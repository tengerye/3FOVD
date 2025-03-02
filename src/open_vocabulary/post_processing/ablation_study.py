import glob
import os
import pickle
import math
from collections import defaultdict
import supervision as sv
import numpy as np
import torch
from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import json


DEVICE = 'cuda:0'

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
        # 'total_scores': sorted_total_scores
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


def preprocess_caption(caption: str, cat: str) -> str:
    caption = cat + "#" + caption
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

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

# 常规nms
def nms(bboxes, scores, iou_thresh):
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
        index = index[idx + 1]  # 处理剩余的边框
    return torch.Tensor(result).long()


def get_grounding_output(model, image, captions, cats_ls, w, h, label_ls, box_threshold=0):
    device = "cuda"
    image = image.to(device)
    # caption 预处理  拼接为 "类别名称#caption" 的格式
    captions = [preprocess_caption(caption, cat) for caption, cat in zip(captions, cats_ls)]
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
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        # 获取最大值对应的索引位置(即对应类别)
        max_index = logits_filt.max(dim=1)[1]

        # 截取目标词汇对应的logits索引
        # 初始化分词器
        tokenizer = model.tokenizer
        # 截取出目标类别词
        test_caption = caption.split("#")[0]
        test_tokenized = tokenizer(test_caption)
        offset = 1
        # 获取目标类别词的序列长度
        token_lens = len(test_tokenized["input_ids"][offset:-1])
        # 获取目标类别词的分数
        target_words_max_logits = logits_filt[:, offset: token_lens + offset].max(dim=1)[0]

        pred_boxes = []
        # 对bbox进行格式转换
        for box in boxes_filt:
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([w, h, w, h])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            pred_boxes.append(box)
        tmp_boxes = torch.stack(pred_boxes, dim=0)

        res_index = nms_self(tmp_boxes, target_words_max_logits, 0.5, max_index)
        tmp_boxes, target_words_max_logits = (torch.index_select(tmp_boxes, dim=0, index=res_index),
                                              torch.index_select(target_words_max_logits, dim=0, index=res_index))

        # 按照目标类别的预测置信度进行过滤
        mask = target_words_max_logits > box_threshold
        tmp_scores = target_words_max_logits[mask]
        tmp_boxes = tmp_boxes[mask]

        for box, score in zip(tmp_boxes, tmp_scores):
            res_pred_boxes.append(box)
            res_pred_scores.append(score)
            res_pred_labels.append(label_id)

    return sort_boxes_by_score(res_pred_boxes, res_pred_labels, res_pred_scores, 1e6)

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
            words = tokenizer.batch_decode([token_id-1, token_id, token_id+1])
            phrases.append(','.join(words))
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
def get_grounding_output_baseline(model, image, captions, cats_ls, w, h, label_ls, iou=0.5, box_threshold=0):
    device = "cuda"
    image = image.to(device)
    captions = [preprocess_caption(caption, cat) for caption, cat in zip(captions, cats_ls)]
    # we need to batch the inputs, in order to make a query for each caption
    images = image[None].repeat(len(captions), 1, 1, 1)

    with torch.no_grad():
        outputs = model(images, captions=captions)


    logits_batched = [x.cpu().sigmoid() for x in outputs['pred_logits']]
    boxes_batched = [x.cpu() for x in outputs['pred_boxes']]

    res_pred_boxes = []
    res_pred_scores = []
    res_pred_labels = []
    for logits, boxes, label_id in zip(logits_batched, boxes_batched, label_ls):

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask].max(dim=1)[0]  # num_filt, 1
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        pred_boxes = []
        for box in boxes_filt:
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([w, h, w, h])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            pred_boxes.append(box)
        tmp_boxes = torch.stack(pred_boxes, dim=0)
        res_index = nms(tmp_boxes, logits_filt, 0.5)
        res_boxes, res_logits = (torch.index_select(tmp_boxes, dim=0, index=res_index),
                                              torch.index_select(logits_filt, dim=0, index=res_index))

        for box, score in zip(res_boxes, res_logits):
            res_pred_boxes.append(box)
            res_pred_scores.append(score)
            res_pred_labels.append(label_id)

    return sort_boxes_by_score(res_pred_boxes, res_pred_labels, res_pred_scores, 1e6)


# IMP1： 消融实验1
def get_grounding_output_IMP1(model, image, captions, cats_ls, image_source, label_ls, box_threshold=0, img_id=0):
    device = DEVICE
    image = image.to(device)
    # caption 预处理  拼接为 "类别名称#caption" 的格式
    captions = [preprocess_caption(caption, cat) for caption, cat in zip(captions, cats_ls)]
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

        # 将预测置信度小于box_threshold的行进行过滤
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        pred_boxes = []
        # 对bbox进行格式转换
        for box in boxes_filt:
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([image_source.shape[1], image_source.shape[0], image_source.shape[1], image_source.shape[0]])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            pred_boxes.append(box)
        tmp_boxes = torch.stack(pred_boxes, dim=0)
        # 截取目标词汇对应的logits索引
        # 初始化分词器
        tokenizer = model.tokenizer
        # 将产出的框和对应的token进行可视化
        draw_grounding_output(tokenizer, caption, tmp_boxes, logits_filt, image_source, f"./image/exp1/test_draw/official_demo/test{img_id}_{label_id}.jpg")
        # 截取出目标类别词
        test_caption = caption.split("#")[0]
        test_tokenized = tokenizer(test_caption)
        offset = 1
        # 获取目标类别词的序列长度
        token_lens = len(test_tokenized["input_ids"][offset:-1])
        # 获取目标类别词的分数bbox
        target_words_max_value, target_words_max_index = logits_filt[:, offset: token_lens + offset].max(dim=1)
        target_index = nms(
            tmp_boxes,
            target_words_max_value,
            iou_thresh=0.5  # TODO: set as parameter
        )

        for idx in target_index:
            if logits_filt[idx, target_words_max_index[idx]] > 0.25:
                logits_filt[idx, target_words_max_index[idx]] = 1.0

        res_index = nms(tmp_boxes, logits_filt.max(dim=1)[0], 0.5)
        res_boxes, res_logits = (torch.index_select(tmp_boxes, dim=0, index=res_index),
                                              torch.index_select(logits_filt, dim=0, index=res_index))

        for box, score in zip(res_boxes, res_logits):
            res_pred_boxes.append(box)
            res_pred_scores.append(score.max().item())
            res_pred_labels.append(label_id)

    return sort_boxes_by_score(res_pred_boxes, res_pred_labels, res_pred_scores, 1e6)


# IMP2
def get_grounding_output_IMP2(model, image, captions, cats_ls, w, h, label_ls, box_threshold=0):
    device = "cuda"
    image = image.to(device)
    # caption 预处理  拼接为 "类别名称#caption" 的格式
    captions = [preprocess_caption(caption, cat) for caption, cat in zip(captions, cats_ls)]
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
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        # 获取最大值对应的索引位置(即对应类别)以及分值(全局最高分值)
        logits_filt, max_index = logits_filt.max(dim=1)

        pred_boxes = []
        # 对bbox进行格式转换
        for box in boxes_filt:
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([w, h, w, h])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            pred_boxes.append(box)
        tmp_boxes = torch.stack(pred_boxes, dim=0)

        res_index = nms_self(tmp_boxes, logits_filt, 0.5, max_index)
        tmp_boxes, logits_filt = (torch.index_select(tmp_boxes, dim=0, index=res_index),
                                              torch.index_select(logits_filt, dim=0, index=res_index))

        for box, score in zip(tmp_boxes, logits_filt):
            res_pred_boxes.append(box)
            res_pred_scores.append(score)
            res_pred_labels.append(label_id)

    return sort_boxes_by_score(res_pred_boxes, res_pred_labels, res_pred_scores, 1e6)

def saveObject(obj, path):
    print("Saving " + path + '.pkl')
    if not os.path.exists(path):
        os.makedirs(path)
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


def gen_predict_data(img_dir, annotation_filepath, BOX_TRESHOLD):
    # 模型加载
    model = load_model("/home/ubuntu/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/ubuntu/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    model = model.to(DEVICE)
    model.eval()
    
    img_ls = glob.glob(f"{img_dir}/*.jpg")
    data = read_json(annotation_filepath)
    img2cat_dict = getImage2CatsMap(data)
    cats_dict = getCatsId2Item(data)
    complete_img_output = []
    img_processed = 0
    zero_bbox_cnt = 0
    for img_path in tqdm(img_ls):
        image_source, image = load_image(img_path)
        filename = img_path.split("/")[-1]
        rel_cat_ids = img2cat_dict[filename]
        caption_ls, label_ls, cats_ls = [], [], []
        for id in rel_cat_ids:
            caption_ls.append(cats_dict[id]["desc"])
            label_ls.append(cats_dict[id]["id"])
            cats_ls.append(cats_dict[id]["name"])
        loop = math.ceil(len(caption_ls) / 10)
        img_output = {
            "boxes": [],
            "labels": [],
            "scores": [],
            "img_path": img_path
        }
        for i in range(loop):
            batch_caption_list = caption_ls[i * 10: (i + 1) * 10]
            batch_label_list = label_ls[i * 10: (i + 1) * 10]
            batch_cats_ls = cats_ls[i * 10: (i + 1) * 10]
            # res = get_grounding_output(model, image, batch_caption_list, cats_ls, image_source.shape[1],
            #                            image_source.shape[0],
            #                            box_threshold=BOX_TRESHOLD, label_ls=batch_label_list)
            res = get_grounding_output_IMP1(model, image, batch_caption_list, batch_cats_ls, image_source,
                                        box_threshold=BOX_TRESHOLD, label_ls=batch_label_list, img_id = filename[:-4] + '_' + str(batch_label_list[i]))
            # res = get_grounding_output_origin(model, image, batch_caption_list, batch_cats_ls, image_source.size[0],
            #                                   image_source.size[1],
            #                                   box_threshold=BOX_TRESHOLD, label_ls=batch_label_list)


            img_output["boxes"].extend(res["boxes"])
            img_output["labels"].extend(res["labels"])
            img_output["scores"].extend(res["scores"])
        print(f"图片{img_path}的候选框总数为: {len(img_output['labels'])}")
        if len(img_output['labels']) == 0:
            zero_bbox_cnt += 1
        img_processed += 1
        complete_img_output.append(img_output)
        # 每一千张图片就存储一次数据
        if img_processed % 1000 == 0 or img_processed == len(img_ls):
            print(f"当前bbox为空的image累积为{zero_bbox_cnt}个")
            print(f"当前已完成{img_processed}张图片")
            # 每1000个图片处理完成后，就生成一个结果文件进行存储
            saveObject(complete_img_output,
                       f"./output/ablation_study/product/exp1/dino_final_{img_processed}")
            # 将存储结果的变量清空，节省内存
            complete_img_output = []


def getPathByEnv(env):
    if env == "vehicle":
        img_dir = "/data/data/vehicle/images_test/"
        annotation_filepath = "/data/chaihaojiang/our_data/GroundingDINO/demo/new_annotations_test4.json"
    else:
        img_dir = "/data/data/product_yolo/test/images/"
        annotation_filepath = "./product_update.json"
    return img_dir, annotation_filepath


def main():
    # img_dir, annotation_filepath = getPathByEnv("vehicle")
    img_dir = '/home/ubuntu/workspace/3FOVD/datasets/3FOVD-V/images'
    annotation_filepath = '/home/ubuntu/workspace/3FOVD/datasets/3FOVD-V/new_annotations_test4.json'
    BOX_TRESHOLD = 0.2
    gen_predict_data(img_dir, annotation_filepath, BOX_TRESHOLD)


if __name__ == '__main__':
    main()
