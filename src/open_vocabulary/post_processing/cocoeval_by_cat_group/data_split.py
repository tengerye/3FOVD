
import json
from itertools import islice
import os

from tqdm import tqdm

CATEGORIES_PER_GROUP = 20  # 每组类别数

def chunk_indices(lst, chunk_size):
    """将列表索引分块"""
    for i in range(0, len(lst), chunk_size):
        yield list(lst[i:i + chunk_size])

def main(detection_path, output_dir):
    # 加载原始检测结果
    with open(detection_path, 'r') as f:
        all_detections = json.load(f)

    # 获取所有唯一类别ID并排序
    all_categories = sorted({d['category_id'] for d in all_detections})

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 分块处理
    for group_idx, cat_group in enumerate(tqdm(chunk_indices(all_categories, CATEGORIES_PER_GROUP))):
        # 筛选当前组的检测结果
        group_detections = [
            d for d in all_detections
            if d['category_id'] in cat_group
        ]

        # 保存分组结果
        output_path = os.path.join(output_dir, f'group_{group_idx}.json')
        with open(output_path, 'w') as f:
            json.dump(group_detections, f)

        print(f'Group {group_idx}: {len(cat_group)} categories, {len(group_detections)} detections')
    pass


if __name__ == '__main__':
    detection_path = "/root/post_process_data/dino/vehicle/groundingdino_vehicle_test_prediction_coco_baseline.json"
    output_dir = "/root/exp/detection_group"
    main(detection_path, output_dir)