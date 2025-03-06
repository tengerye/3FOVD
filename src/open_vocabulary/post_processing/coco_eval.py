#!/usr/bin/env python3

import json
import time
from datetime import datetime
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_nowtime():
    now = datetime.now()
    formatted_time = now.strftime('%Y/%m/%d %H:%M:%S')
    return formatted_time


def coco_evaluation(gt_json_path, pred_json_path, iou_type='bbox'):
    """
    Evaluate object detection results on a COCO-style dataset.

    :param gt_json_path: Path to COCO-style ground truth JSON.
    :param pred_json_path: Path to JSON with detection results.
    :param iou_type: The iouType to evaluate ('bbox', 'segm', or 'keypoints').
    """
    print(f"{get_nowtime()} 开始加载数据")
    # Load ground truth annotations
    coco_gt = COCO(gt_json_path)

    # Load predicted results
    coco_dt = coco_gt.loadRes(pred_json_path)

    print(f"{get_nowtime()}  数据加载完毕")
    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    print(f"{get_nowtime()} 开始evaluate")
    # Run evaluation
    coco_eval.evaluate()
    print(f"{get_nowtime()} 结束evaluate")
    print(f"{get_nowtime()} 开始accumulate")
    coco_eval.accumulate()
    print(f"{get_nowtime()} 结束accumulate")
    coco_eval.summarize()

    # The COCOeval object holds all evaluation results in coco_eval.stats
    # By default, coco_eval.summarize() prints the metrics in the console.
    # You can also store them in a variable for further processing:
    metrics = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR1": coco_eval.stats[6],
        "AR10": coco_eval.stats[7],
        "AR100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }

    return metrics


if __name__ == "__main__":
    start_time = time.time()
    # Example usage:
    ground_truth_json = "/root/post_process_data/vild/product/instances_product_test.json"
    predictions_json = "/root/post_process_data/dino/groundingdino_product_test_prediction_remove_cover_v2.json"
    # ground_truth_json = "/data/data/final/product/product_yolo/valid/annotations/instances_product_valid.json"
    # predictions_json = "/data/chaihaojiang/postprocess_data_0305/vild/vild_product_val_prediction_visualized_coco_baseline.json"
    results = coco_evaluation(ground_truth_json, predictions_json, iou_type="bbox")
    end_time = time.time()
    print(f"共耗时{end_time - start_time}秒")
    print("COCO Evaluation Results:")
    print(results)
