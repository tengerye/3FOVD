#!/usr/bin/env python3

import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def coco_evaluation(gt_json_path, pred_json_path, iou_type='bbox'):
    """
    Evaluate object detection results on a COCO-style dataset.

    :param gt_json_path: Path to COCO-style ground truth JSON.
    :param pred_json_path: Path to JSON with detection results.
    :param iou_type: The iouType to evaluate ('bbox', 'segm', or 'keypoints').
    """
    # Load ground truth annotations
    coco_gt = COCO(gt_json_path)
    
    # Load predicted results
    coco_dt = coco_gt.loadRes(pred_json_path)
    
    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
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
    # Example usage:
    ground_truth_json = "/home/ubuntu/workspace/3FOVD/datasets/3FOVD-RP/test/instances_rp_test.json"
    predictions_json = "/home/ubuntu/workspace/3FOVD/datasets/3FOVD-RP/test/vild_product_test_prediction_coco_cb.json"

    results = coco_evaluation(ground_truth_json, predictions_json, iou_type="bbox")
    print("COCO Evaluation Results:")
    print(results)
