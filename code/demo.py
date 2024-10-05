import datetime
import os
import json
import time

import numpy as np
import torch
import torchvision

from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate, predict_original, predict_with_nms, predict_not_with_nms
import cv2
from load_test_data import init, det_parent_path, gt_parent_path


def get_classname_by_caption(caption, class_caption_dict):
    for k, v in class_caption_dict.items():
        if v == caption:
            return k


def get_imageobj_by_imagepath(path, image_obj_list):
    for image in image_obj_list:
        if image.img_path == path:
            return image



if __name__ == '__main__':
    # load test data
    test_data, _, class_caption_dict = init()
    # # ======================
    class_list = list(class_caption_dict.keys())
    total_class_num = class_list.__len__()
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
            caption_index += 1
        complete_img_cnt += 1
        complete_imgs.append(IMAGE_PATH)
        # if complete_img_cnt == 3:
        #     break
        log1 = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} the number of complete image is {complete_img_cnt}, the current image is : {IMAGE_PATH}"
        with open(res_path, "a+") as f:
            f.write(f"{log1} \n")
    t2 = time.time()
    log2 = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 运行结束，所耗费的时间为{t2 - t1}"
    with open(res_path, "a+") as f:
        f.write(f"{log2} \n")


























