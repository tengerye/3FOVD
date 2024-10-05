import datetime
import os
import json
import time

import numpy as np
import torch
import torchvision

from torchvision.ops import box_convert
from vo import MAP
from groundingdino.util.inference import load_model, load_image, predict, annotate, predict_original, predict_with_nms, predict_not_with_nms
from test_data import test_img_caption_dict
import cv2
import configparser
from test_data2 import init, det_parent_path, gt_parent_path, init2


def get_classname_by_caption(caption, class_caption_dict):
    for k, v in class_caption_dict.items():
        if v == caption:
            return k


def get_imageobj_by_imagepath(path, image_obj_list):
    for image in image_obj_list:
        if image.img_path == path:
            return image


# # 计算预测框和真实框之间的iou
# def calc_iou(boxes, true_box_list, iou_threshold):
#     cur_box = 1
#     d = {}
#     is_positive = False
#
#     if boxes.size()[0] == 0:  # 一个框都没有
#         return d, is_positive
#     for box in boxes:
#         temp_ls = []
#         center_x = box[0]
#         center_y = box[1]
#         width = box[2]
#         height = box[3]
#         # 转化为 [xyxy]坐标
#         left_x = center_x - width / 2
#         left_y = center_y - height / 2
#         right_x = center_x + width / 2
#         right_y = center_y + height / 2
#         # 计算与每一个true_box的iou
#         for true_box in true_box_list:
#             true_left_x = true_box[0][0]
#             true_left_y = true_box[0][1]
#             true_right_x = true_box[1][0]
#             true_right_y = true_box[1][1]
#
#             x1 = max(left_x, true_left_x)
#             y1 = max(left_y, true_left_y)
#             x2 = min(right_x, true_right_x)
#             y2 = min(right_y, true_right_y)
#
#             overlap_area = (x2 - x1) * (y2 - y1)
#             box_area = width * height
#             true_box_area = (true_right_x - true_left_x) * (true_right_y - true_left_y)
#             iou = overlap_area / (box_area + true_box_area - overlap_area)
#             temp_ls.append(iou)
#         d[cur_box] = temp_ls
#         cur_box += 1
#     iou_list = list(d.values())[0]
#     for iou in iou_list:
#         if iou > iou_threshold:
#             is_positive = True
#     return d, is_positive


# def updateClassConfutionMatrix(class_confusion_matrix, is_positive, is_cur_class, index):
#     if is_positive:
#         if is_cur_class:
#             class_confusion_matrix[index][0] += 1
#         else:
#             class_confusion_matrix[index][1] += 1
#     else:
#         if is_cur_class:
#             class_confusion_matrix[index][3] += 1
#         else:
#             class_confusion_matrix[index][2] += 1


if __name__ == '__main__':
    # test_data, class_caption_dict = init2()
    # # ======================
    # class_list = list(class_caption_dict.keys())
    # total_class_num = class_list.__len__()
    model = load_model("../groundingdino/config/GroundingDINO_SwinT_OGC.py",
                       "../weights/groundingdino_swint_ogc.pth", device="gpu")
    # class_confusion_matrix_iou_list = [[[0, 0, 0, 0] for _ in range(total_class_num)] for _ in range(9)]   # [True Positive, False Positive, True Negative, False Negative]

    BOX_TRESHOLD = 0.25  # 0.35
    TEXT_TRESHOLD = 0.1
    complete_imgs = []
    complete_img_cnt = 0
    res_path = f"./result/run.log"
    t1 = time.time()
    for IMAGE_PATH, captions in test_data.items():
        caption_index = 1
        filename = IMAGE_PATH.split("/")[-1].split(".")[0] + ".txt"
        for caption in captions:
            image_source, image = load_image(IMAGE_PATH)
            h, w, _ = image_source.shape
            # 14
            # boxes, logits, phrases = predict(
            #     model=model,
            #     image=image,
            #     image_source=image_source,
            #     caption=caption,
            #     box_threshold=BOX_TRESHOLD,
            #     text_threshold=TEXT_TRESHOLD
            # )
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                image_source=image_source,
                caption=caption,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            if boxes is None:
                continue
            obj_name = IMAGE_PATH.split("/")[-1].split(".")[0]
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            # 获取当前case的类别信息
            cur_classname = get_classname_by_caption(caption, class_caption_dict)
            class_ = class_list.index(cur_classname)
            path = f"./result/map/{det_parent_path}/{filename}"
            path2 = f"./result/map/{gt_parent_path}/{filename}"
            # 将detect box的内容写入到文件
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxys = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            print(f"==============xyxys的坐表為：f{xyxys}==============")
            if xyxys.shape[0] >= 1:
                for idx in range(len(xyxys)):
                    xyxy = xyxys[idx]
                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    temp_score = logits[0].item()
                    with open(path, "a+") as f:
                        f.write(f"{class_} {logits[idx].item():.2f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
                    open(path2, 'a+').close()
            if class_ in [29, 23, 70, 27, 20]:
                dis_path = f"./result/new_result_0915/{obj_name}_image_shop_{caption_index}_{class_}.jpg"
                cv2.imwrite(dis_path, annotated_frame)
            print(f"======================{obj_name}的第{caption_index}个caption处理完成===================")
            print(f"box的数量一共为{len(boxes)}")
            caption_index += 1
        complete_img_cnt += 1
        complete_imgs.append(IMAGE_PATH)
        # if complete_img_cnt == 3:
        #     break
        log1 = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} the number of complete image is {complete_img_cnt}, the current image is : {IMAGE_PATH}"
        print(log1)
        with open(res_path, "a+") as f:
            f.write(f"{log1} \n")
    t2 = time.time()
    log2 = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 运行结束，所耗费的时间为{t2 - t1}"
    print(log2)
    with open(res_path, "a+") as f:
        f.write(f"{log2} \n")

























    # # 计算map值
    # obj = get_imageobj_by_imagepath(IMAGE_PATH, image_obj_list)
    # # 正确的对比box列表
    # true_box_list = []
    # true_class_list = obj.class_en_list
    # for json in obj.box_info:
    #     true_box = json['points']
    #     true_box_list.append(true_box)
    # is_cur_class = True if cur_classname in true_class_list else False
    # for i in range(1, 10):
    #     predict_box = boxes * torch.Tensor([w, h, w, h])
    #     res_dict, is_positive = calc_iou(predict_box, true_box_list, iou_threshold=i/10)
    #     updateClassConfutionMatrix(class_confusion_matrix_iou_list[i - 1], is_positive, is_cur_class, index)


    # 运行map进行计算
    # det_path = "./result/map/det_data/"
    # groundTruth_path = "./result/map/gt_data/"
    # iou = 0.5
    # points = 0
    # confidence = 0.1
    # model_name = "./result/map/model.names"
    # output_path = "./result/map/output/results_map.json"
    # calculator = MAP(
    #     detPath=det_path,
    #     gtPath=groundTruth_path,
    #     iou=iou,
    #     points=points,
    #     names=model_name,
    #     confidence=confidence,
    #     logName=output_path,
    # )
    #
    # calculator.calcmAP()
    # end_time = time.time()
    # print(f"一共需要{end_time - start_time}秒")





    # # 按照iou维度和class维度计算ap
    # class_precision_list = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(total_class_num)]   # (total_class_num, 9)
    # class_recall_list = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(total_class_num)]
    # for i in range(len(class_confusion_matrix_iou_list)):
    #     class_confusion_matrix_list = class_confusion_matrix_iou_list[i]
    #     for index in range(len(class_confusion_matrix_list)):
    #         class_confusion_matrix = class_confusion_matrix_list[index]
    #         TP = class_confusion_matrix[0]
    #         FP = class_confusion_matrix[1]
    #         TN = class_confusion_matrix[2]
    #         FN = class_confusion_matrix[3]
    #         epsilon = 1e-10  # 小的常数值，确保分母非零
    #         FP = torch.clamp(torch.tensor(FP), min=epsilon)
    #         FN = torch.clamp(torch.tensor(FN), min=epsilon)
    #         precision = TP / (TP + FP)
    #         recall = TP / (TP + FN)
    #         class_precision_list[index][i] = precision
    #         class_recall_list[index][i] = recall
    # aps = []
    # for i in range(len(class_precision_list)):
    #     precision = np.array(class_precision_list[i])
    #     recall = np.array(class_recall_list[i])
    #     recall_sorted_indices = np.argsort(recall)
    #     recall = recall[recall_sorted_indices]
    #     precision = precision[recall_sorted_indices]
    #     ap = np.trapz(precision, recall)
    #     aps.append(ap)
    #     # 输出信息
    #     print(f"类别为{class_list[i]}的ap值为: {ap}")
    # # 计算map
    # map = sum(aps) / len(aps)
    # print(f"map的值为： {map}")



# IMAGE_PATH
# MyData_Car_IMAGE_PATH = "../weights/my_data/car.jpeg"
# MyData_Food_IMAGE_PATH = "../weights/my_data/food.jpeg"
# EtcData_black_cat_IMAGE_PATH = "../weights/etc_data/black_cat.jpg"
# EtcData_logo_IMAGE_PATH = "../weights/etc_data/logo.jpeg"
# EtcData_plate_IMAGE_PATH = "../weights/etc_data/plate.jpeg"
#
# # ============ caption ===================
# car_caption1 = "The headlights are connected oblique inverted trapezoid or connected oblique paddle type or connected oblique parallelogram or split oblique oval, Volkswagen logo, two or four horizontal bar grille texture, the hood on both sides of the slit line or no slit line, the overall outline of the body is smooth."
# car_caption2 = "The headlights of the car are split diagonal narrow boat light or split diagonal wide boat narrow light, Chinese car logo, flying wing type double kidney air intake grille vertical bar grille texture, trapezoidal fog lamp cover plate, the hood on both sides of the fold line protrudates slightly flat overall, the overall outline is smooth."
# car_caption3 = "Zhonghua H530 sedan#the headlights of the car are split diagonal narrow boat light or split diagonal wide boat narrow light, Chinese car logo, flying wing type double kidney air intake grille vertical bar grille texture, trapezoidal fog lamp cover plate, the hood on both sides of the fold line protrudates slightly flat overall, the overall outline is smooth."
# car_caption4 = "Roewe RX5 SUV, the headlights are conjoined diagonally long parallelogram or conjoined diagonally inverted trapezoidal pierced by the grille, Roewe logo, inverted trapezoidal banner grille or inverted trapezoidal pure black mesh grille, quadrilateral fog lamps or figure-7 fog lamps, there are polygonal protrusions on both sides of the hood, the whole has a little curvature, and the overall contour of the body is smooth, typical SUV models."
#
# demo_caption = "Volkswagen Sagitar sedan, the headlights are connected oblique inverted trapezoid or connected oblique paddle type or connected oblique parallelogram or split oblique oval, Volkswagen logo, two or four horizontal bar grille texture, the hood on both sides of the slit line or no slit line, the overall outline of the body is smooth."
#
# food_caption1 = 'Passion fruit Dove chocolates are packaged in rectangular plastic bags. The middle is "DOVE", the top is "Dove", the bottom is "passion fruit white chocolate", the right is the passion fruit illustration, the bottom right is the net content and "picture for reference only".'
# food_caption2 = 'Puff cookies are packaged in yellow cuboid paper boxes. The top part is printed with "Aji ", the middle part of the package is printed with" puff cookies "in white on a red background, the package is printed around the puff pattern, and the lower left corner is the taste and net content of the product.'
# food_caption3 = 'Coke is packaged in red cylindrical iron cans. On the top of the package are the words "Coca-Cola" and below them are "Classic delicious". At the bottom right of the package is "soda" and the corresponding net content and energy information.'
#
# black_cat_caption1 = 'A dark green fabric pillow.'
# black_cat_caption2 = 'A dark green crochet pillow.'
# black_cat_caption3 = 'A dark orange fabric pillow.'
# black_cat_caption4 = 'A brown fabric pillow.'
# black_cat_caption5 = 'A dark red wood pillow.'
#
# logo_caption1 = 'A grey remote control with a black plastic logo.'
# logo_caption2 = 'A pink remote control with a black plastic logo.'
# logo_caption3 = 'A light yellow remote control with a black plastic logo.'
#
# plate_caption1 = 'A white ceramic plate.'
# plate_caption2 = 'A white stone plate.'
# plate_caption3 = 'A dark blue ceramic plate.'
# plate_caption4 = 'A green wool plate.'


# test_img_caption_dict = {
#         # MyData_Car_IMAGE_PATH: [car_caption1, car_caption2],
#         # MyData_Car_IMAGE_PATH: [car_caption3]
#         # MyData_Food_IMAGE_PATH: [food_caption1, food_caption2, food_caption3]
#         # EtcData_black_cat_IMAGE_PATH: [black_cat_caption1, black_cat_caption2, black_cat_caption3
#         #                                , black_cat_caption4, black_cat_caption5],
#         # EtcData_logo_IMAGE_PATH: [logo_caption1, logo_caption2, logo_caption3],
#         # EtcData_plate_IMAGE_PATH: [plate_caption4] # plate_caption1, plate_caption2, plate_caption3,
#     }
