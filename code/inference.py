from typing import Tuple, List

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert, batched_nms, nms
import bisect

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap


# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


# nms做合并
def nms_self(bboxes, scores, iou_thresh, img_shape, max_score_index_list):
    """

    :param bboxes: 检测框列表
    :param scores: 置信度列表
    :param iou_thresh: IOU阈值
    :return:
    """
    img_h, img_w = img_shape[0], img_shape[1]

    # 将[x, y, w, h] 转化为 [x, y, x, y]
    center_point_x = bboxes[:, 0] * img_w
    center_point_y = bboxes[:, 1] * img_h
    width = bboxes[:, 2] * img_w
    height = bboxes[:, 3] * img_h
    x1 = center_point_x - width / 2
    y1 = center_point_y - height / 2  # 左下角的坐标
    x2 = center_point_x + width / 2
    y2 = center_point_y + height / 2  # 右上角的坐标
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
        # if 0.28 < scores[i] < 0.30:
        # 计算其他边界框与该边界框的IOU
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11)
        h = np.maximum(0, y22 - y11)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # 只保留满足IOU阈值的索引
        idx = np.where(ious <= iou_thresh)[0]
        # 将满足iou，且位置大小完全被当前置信度最高的框包含的小框过滤
        mask_index = []
        value_map = []
        for k in range(len(idx)):
            no = idx[k]
            original_index = index[no + 1]
            if x1[original_index] > x1[i] and y1[original_index] > y1[i] and x2[original_index] < x2[i] and y2[original_index] < y2[i]:
                # 将完全重合的元素进行删除
                mask_index.append(k)
                value_map.append(no)
        idx = np.delete(idx, mask_index)
        # 将不满足IOU阈值的索引合并 （不同类别的候选框进行合并）
        not_match_ids = []
        for j in range(len(ious)):
            cur_class_index = index[j + 1]
            cur_class = max_score_index_list[cur_class_index]
            if (j not in idx) and (j not in value_map) and (cur_class != main_class):
                not_match_ids.append(j)
        # 计算合并的区域
        x1_l, x2_l, y1_l, y2_l = [x1[i].item()], [x2[i].item()], [y1[i].item()], [y2[i].item()]
        for i2 in not_match_ids:
            index2 = index[i2 + 1]
            x1_l.append(x1[index2].item())
            x2_l.append(x2[index2].item())
            y1_l.append(y1[index2].item())
            y2_l.append(y2[index2].item())
        x11 = min(x1_l)
        y11 = min(y1_l)
        x22 = max(x2_l)
        y22 = max(y2_l)
        width = x22 - x11
        height = y22 - y11
        bboxes[i, 0] = (x11 + width / 2.0) / img_w
        bboxes[i, 1] = (y11 + height / 2.0) / img_h
        bboxes[i, 2] = width / img_w
        bboxes[i, 3] = height / img_h
        index = index[idx + 1]  # 处理剩余的边框
    res_index = torch.tensor(result)
    # bboxes, scores = torch.index_select(bboxes, dim=0, index=res_index), torch.index_select(scores, dim=0, index=res_index)
    return res_index


# def nms_self_without_merge(bboxes, scores, iou_thresh, img_shape):
#     """
#
#     :param bboxes: 检测框列表
#     :param scores: 置信度列表
#     :param iou_thresh: IOU阈值
#     :return:
#     """
#     img_h, img_w = img_shape[0], img_shape[1]
#
#     # 将[x, y, w, h] 转化为 [x, y, x, y]
#     center_point_x = bboxes[:, 0] * img_w
#     center_point_y = bboxes[:, 1] * img_h
#     width = bboxes[:, 2] * img_w
#     height = bboxes[:, 3] * img_h
#     x1 = center_point_x - width / 2
#     y1 = center_point_y - height / 2  # 左下角的坐标
#     x2 = center_point_x + width / 2
#     y2 = center_point_y + height / 2  # 右上角的坐标
#     areas = (y2 - y1) * (x2 - x1)
#
#     # 结果列表
#     result = []
#     index = torch.flip(scores.argsort(), [0])  # 对检测框按照置信度进行从高到低的排序，并获取索引
#     # 下面的操作为了安全，都是对索引处理
#     while len(index) > 0:
#         # 当检测框不为空一直循环
#         i = index[0]
#         result.append(i)  # 将置信度最高的加入结果列表
#         # if 0.28 < scores[i] < 0.30:
#         # 计算其他边界框与该边界框的IOU
#         x11 = np.maximum(x1[i], x1[index[1:]])
#         y11 = np.maximum(y1[i], y1[index[1:]])
#         x22 = np.minimum(x2[i], x2[index[1:]])
#         y22 = np.minimum(y2[i], y2[index[1:]])
#         w = np.maximum(0, x22 - x11)
#         h = np.maximum(0, y22 - y11)
#         overlaps = w * h
#         ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
#         # 只保留满足IOU阈值的索引
#         idx = np.where(ious <= iou_thresh)[0]
#         index = index[idx + 1]  # 处理剩余的边框
#     res_index = torch.tensor(result)
#     # bboxes, scores = torch.index_select(bboxes, dim=0, index=res_index), torch.index_select(scores, dim=0, index=res_index)
#     return res_index


# def apply_NMS(boxes, scores, iou=0.5):
#     # indexes_to_keep = batched_nms(boxes,
#     #                               torch.FloatTensor(scores),
#     #                               torch.IntTensor([0] * len(boxes)),
#     #                               iou)
#     indexes_to_keep = nms_self(boxes, torch.FloatTensor(scores), iou)
#     filtered_boxes = []
#     filtered_scores = []
#     # filtered_labels = []
#     deleted_boxes = []
#     deleted_scores = []
#     deleted_labels = []
#
#     for x in range(len(boxes)):
#         if x in indexes_to_keep:
#             filtered_boxes.append(boxes[x])
#             filtered_scores.append(scores[x])
#             # filtered_labels.append(labels[x])
#         else:
#             deleted_boxes.append(boxes[x])
#             deleted_scores.append(scores[x])
#             # deleted_labels.append(labels[x])
#
#     # bisognerebbe fare in modo che le box eliminate abbiano gli score appesi a quelle rimaste
#     # return filtered_boxes, filtered_scores, filtered_labels, create_total_scores(filtered_boxes, filtered_scores, filtered_labels, deleted_boxes, deleted_scores, deleted_labels)
#
#     return filtered_boxes, filtered_scores # , filtered_labels

# 最终版本
def predict(
        model,
        image: torch.Tensor,
        image_source,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
):
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    # 获取模型的原始输出
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
    # 按照最大值进行阈值过滤
    logits, _ = prediction_logits.max(dim=1)
    mask = logits > box_threshold
    temp_logits = prediction_logits[mask]  # temp_logits.shape = (n, 256)
    _, max_index = temp_logits.max(dim=1)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    # 截取目标词汇对应的logits索引
    # 初始化分词器
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    # 截取出目标词
    test_caption = caption.split("#")[0]
    test_tokenized = tokenizer(test_caption)
    offset = 1
    # 获取目标单词的序列长度
    token_lens = len(test_tokenized["input_ids"][offset:-1])
    target_words_max_logits = temp_logits[:, offset: token_lens+offset].max(dim=1)[0]

    # 传入nms，获取过滤结束后的索引，根据索引从原始的logits和、boxes中将结果取出
    h, w, _ = image_source.shape
    res_index = nms_self(boxes, target_words_max_logits, 0.5, [h, w], max_index)  # logits.shape = (nq, 1)
    # 按照nms的结果对原始的prediction_logits 和 prediction_boxes进行过滤
    if res_index.shape[0] == 0:
        return None, None, None
    boxes, logits = torch.index_select(boxes, dim=0, index=res_index), torch.index_select(temp_logits, dim=0,
                                                                                              index=res_index)
    # 将nms过滤后剩余的框，根据目标类别置信度进行过滤
    # 截取 logits 张量的前token_lens列
    logits = logits[:, offset:token_lens+offset]
    # 从截取后的 logits 中提取每个预测类别的得分，并找到最大值
    max_scores, _ = logits.max(dim=1)
    # 根据边框的置信度阈值过滤得分
    mask = max_scores > box_threshold
    # 根据过滤条件提取对应的 logits 和边界框
    logits = logits[mask]
    boxes = boxes[mask]

    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]

        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.',
                                                                                                                   ''))
    else:

        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, offset).replace('.', '')
            for logit
            in logits
        ]

    logits = logits.max(dim=1)[0]
    return boxes, logits, phrases

# 做nms并进行合并
def predict_with_nms(
        model,
        image: torch.Tensor,
        image_source,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
):
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    # 获取模型的原始输出
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
    # 按照最大值进行阈值过滤
    logits, _ = prediction_logits.max(dim=1)
    mask = logits > box_threshold
    temp_logits = prediction_logits[mask]  # temp_logits.shape = (n, 256)
    _, max_index = temp_logits.max(dim=1)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    # 截取目标词汇对应的logits索引
    # 初始化分词器
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    # 截取出目标词
    test_caption = caption.split("#")[0]
    test_tokenized = tokenizer(test_caption)
    offset = 1
    # 获取目标单词的序列长度
    token_lens = len(test_tokenized["input_ids"][offset:-1])
    target_words_max_logits = temp_logits[:, offset: token_lens + offset].max(dim=1)[0]


    # 传入nms，获取过滤结束后的索引，根据索引从原始的logits和、boxes中将结果取出
    h, w, _ = image_source.shape
    res_index = nms_self(boxes, target_words_max_logits, 0.5, [h, w], max_index)  # logits.shape = (nq, 1)
    # 按照nms的结果对原始的prediction_logits 和 prediction_boxes进行过滤
    if res_index.shape[0] == 0:
        return None, None, None
    boxes, logits = torch.index_select(boxes, dim=0, index=res_index), torch.index_select(temp_logits, dim=0,
                                                                                              index=res_index)
    # # 将nms过滤后剩余的框，根据目标类别置信度进行过滤
    # # 截取 logits 张量的前token_lens列
    # logits = logits[:, offset:token_lens+offset]
    # # 从截取后的 logits 中提取每个预测类别的得分，并找到最大值
    # max_scores, _ = logits.max(dim=1)
    # # 根据边框的置信度阈值过滤得分
    # mask = max_scores > box_threshold
    # # 根据过滤条件提取对应的 logits 和边界框
    # logits = logits[mask]
    # boxes = boxes[mask]

    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]

        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.',
                                                                                                                   ''))
    else:

        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, offset).replace('.', '')
            for logit
            in logits
        ]

    logits = logits.max(dim=1)[0]
    return boxes, logits, phrases


# 原始版本：只做最大值过滤
def predict_original(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (5, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (5, 4)
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]

        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.',
                                                                                                                   ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, offset=0).replace('.', '')
            for logit
            in logits
        ]

    return boxes, logits.max(dim=1)[0], phrases

# 不进行nms过滤，使用前三个单词置信度进行过滤
def predict_not_with_nms(
        model,
        image: torch.Tensor,
        image_source,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
):
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    # 获取模型的原始输出
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
    # 按照最大值进行阈值过滤
    logits, _ = prediction_logits.max(dim=1)
    mask = logits > box_threshold
    temp_logits = prediction_logits[mask]  # temp_logits.shape = (n, 256)
    _, max_index = temp_logits.max(dim=1)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    # 截取目标词汇对应的logits索引
    # 初始化分词器
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    # 截取出目标词
    test_caption = caption.split("#")[0]
    test_tokenized = tokenizer(test_caption)
    offset = 1
    # 获取目标单词的序列长度
    token_lens = len(test_tokenized["input_ids"][offset:-1])
    # target_words_max_logits = temp_logits[:, offset: token_lens+offset].max(dim=1)[0]

    # # 传入nms，获取过滤结束后的索引，根据索引从原始的logits和、boxes中将结果取出
    # h, w, _ = image_source.shape
    # res_index = nms_self(boxes, target_words_max_logits, 0.5, [h, w], max_index)  # logits.shape = (nq, 1)
    # # 按照nms的结果对原始的prediction_logits 和 prediction_boxes进行过滤
    # if res_index.shape[0] == 0:
    #     return None, None, None
    # boxes, logits = torch.index_select(boxes, dim=0, index=res_index), torch.index_select(temp_logits, dim=0,
    #                                                                                           index=res_index)
    # 据目标类别置信度进行过滤
    # 截取 logits 张量的前token_lens列
    # if len(logits.size()) == 1:
    #     logits = logits[offset:token_lens+offset]
    #     logits = torch.reshape(logits, (1, -1))
    #     max_scores, _ = logits.max(dim=1)
    #     # 根据边框的置信度阈值过滤得分
    #     mask = max_scores > box_threshold
    #     if not mask:
    #         return None, None, None
    # else:
    logits = temp_logits[:, offset:token_lens + offset]
    # 从截取后的 logits 中提取每个预测类别的得分，并找到最大值
    max_scores, _ = logits.max(dim=1)
    # 根据边框的置信度阈值过滤得分
    mask = max_scores > box_threshold
    # 根据过滤条件提取对应的 logits 和边界框
    logits = logits[mask]
    boxes = boxes[mask]

    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]

        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.',
                                                                                                                   ''))
    else:

        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, offset).replace('.', '')
            for logit
            in logits
        ]
    logits = logits.max(dim=1)[0]
    return boxes, logits, phrases
# 做nms并不进行合并
# def predict_with_nms_not_merge(
#         model,
#         image: torch.Tensor,
#         image_source,
#         caption: str,
#         box_threshold: float,
#         text_threshold: float,
#         device: str = "cuda",
#         remove_combined: bool = False
# ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
#     caption = preprocess_caption(caption=caption)
#
#     model = model.to(device)
#     image = image.to(device)
#
#     with torch.no_grad():
#         outputs = model(image[None], captions=[caption])
#
#     prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
#     prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
#     # 阈值过滤
#     logits, _ = prediction_logits.max(dim=1)  # 最大值降维
#     mask = logits > box_threshold
#     logits = logits[mask]  # logits.shape = (n, 256)
#     boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
#     # 传入nms,
#     h, w, _ = image_source.shape
#     res_index = nms_self_without_merge(boxes, logits, 0.1, [h, w])  # logits.shape = (nq, 1)
#     temp_logits = prediction_logits[mask]
#     # 按照nms的结果对原始的prediction_logits 和 prediction_boxes进行过滤
#     boxes, logits = torch.index_select(boxes, dim=0, index=res_index), torch.index_select(temp_logits, dim=0,
#                                                                                           index=res_index)
#     print(f"nms处理完成之后的boxes的数量为{len(boxes)}")
#     # boxes = torch.stack(boxes, dim=0)
#     # # 升维
#     # logits = torch.tensor(logits)
#     # logits = logits.unsqueeze(-1)
#     # lines = logits.shape[0]
#     # ls = []
#     # for index in range(lines):
#     #     col = logits[index]
#     #     idx = max_index[index]
#     #     new_elem = torch.zeros(255)
#     #     new_col = torch.cat((col, new_elem), dim=0)
#     #     # 将最大值回归原始位置
#     #     new_col[0], new_col[idx] = 0, col[0]
#     #     ls.append(new_col)
#     # logits = torch.stack(ls)
#
#     tokenizer = model.tokenizer
#     tokenized = tokenizer(caption)
#     if remove_combined:
#         sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
#
#         phrases = []
#         for logit in logits:
#             max_idx = logit.argmax()
#             insert_idx = bisect.bisect_left(sep_idx, max_idx)
#             right_idx = sep_idx[insert_idx]
#             left_idx = sep_idx[insert_idx - 1]
#             phrases.append(
#                 get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.',
#                                                                                                                    ''))
#     else:
#         phrases = [
#             get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
#             for logit
#             in logits
#         ]
#
#     # return boxes, logits.max(dim=1)[0], phrases
#     logits = logits.max(dim=0)[0]
#     return boxes, logits, phrases





def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)
    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]


    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
            self,
            model_config_path: str,
            model_checkpoint_path: str,
            device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
            self,
            image: np.ndarray,
            caption: str,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
            self,
            image: np.ndarray,
            classes: List[str],
            box_threshold: float,
            text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)
