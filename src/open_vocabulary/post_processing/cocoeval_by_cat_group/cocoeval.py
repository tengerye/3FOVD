# evaluate_group.py
import json
import os
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

def main(group_idx, detection_dir, groundtruth_path, output_path):

    # 加载检测结果
    det_path = os.path.join(detection_dir, f'group_{group_idx}.json')
    with open(det_path, 'r') as f:
        detections = json.load(f)

    # 加载COCO真实标注
    coco_gt = COCO(groundtruth_path)

    # 加载检测结果到COCO格式
    coco_dt = coco_gt.loadRes(detections)

    # 获取当前组包含的类别
    cat_ids = list({d['category_id'] for d in detections})

    # 存储每个类别的AP
    category_aps = {}

    # 逐类别评估
    for cat_id in tqdm(cat_ids):
        # 初始化评估器
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = [cat_id]

        # 评估过程
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 记录AP（IoU 0.5:0.95）
        category_aps[int(cat_id)] = coco_eval.stats[0].item()  # 转换为Python float

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(category_aps, f)


if __name__ == '__main__':
    group_idx = int(sys.argv[1])
    groundtruth_path = "/root/post_process_data/dino/vehicle/instances_vehicle_test.json"
    detection_dir = "/root/exp/detection_group"
    output_path = f"/root/exp/aps/group_{group_idx}_ap.json"
    main(group_idx, detection_dir, groundtruth_path, output_path)