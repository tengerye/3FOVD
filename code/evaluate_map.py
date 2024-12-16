import datetime
import json
import pickle
import glob
from collections import defaultdict
from typing import Union, Dict, List
import argparse
import torch
from torch import Tensor, IntTensor
from tqdm import tqdm

try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP
    MeanAveragePrecision = MAP

from demo import read_json

DEVICE = 'cpu'

def transform_predslist_to_dict(preds):
    result = {}
    for pred in preds:
        image = pred['image_path'].split("/")[-1]
        if image not in result:
            result[image] = []
        result[image].append(pred)
    return result

def convert_format(boxes, dim=2):
    if dim == 2:
        for box in boxes:
            box[2] += box[0]
            box[3] += box[1]
    else:
        boxes[2] += boxes[0]
        boxes[3] += boxes[1]
    return boxes
def assert_box(boxes, dim = 2):
    """Check that the box is in [xmin, ymin, xmax, ymax] format"""
    if dim == 2:
        for box in boxes:
            assert box[0] <= box[2] and box[1] <= box[3]
    else:
        assert boxes[0] <= boxes[2] and boxes[1] <= boxes[3]

def get_image_ground_truth(data, image_id):
    image_data = {'boxes': [], 'labels': []}

    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            image_data['boxes'].append(annotation['bbox'])
            image_data['labels'].append(annotation['category_id'])

    image_data['boxes'] = convert_format(image_data['boxes'])
    assert_box(image_data['boxes'])
    image_data['boxes'] = Tensor(image_data['boxes']).to(DEVICE)
    image_data['labels'] = IntTensor(image_data['labels']).to(DEVICE)

    return image_data

def get_all_image_ground_truth(data):
    res = {-1: {'boxes': [], 'labels': []}}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id not in res:
            res[image_id] = {'boxes': [], 'labels': []}
        box = convert_format(annotation['bbox'], dim=1)
        assert_box(box, dim=1)
        res[image_id]['boxes'].append(box)
        res[image_id]['labels'].append(annotation['category_id'])
    # 格式处理
    for k, v in res.items():
        v['boxes'] = Tensor(v['boxes']).to(DEVICE)
        v['labels'] = IntTensor(v['labels']).to(DEVICE)
    return res

def get_image_preds(preds):
    labels = []
    scores = []
    boxes = []
    for pred in preds:
        labels += [x for x in pred['labels']]
        scores += [x for x in pred['scores']]
        boxes += ([x for x in pred['boxes']])
        assert_box(boxes)
    if boxes != []:
        boxes = boxes
    else:
        boxes = [[0, 0, 0, 0]]
        scores = [0]
        labels = [-10]
    # boxes = boxes if boxes != [] else [[0, 0, 0, 0]]
    if type(boxes[0]) != torch.Tensor:
        return {
            'boxes': Tensor(boxes).to(DEVICE),
            'labels': IntTensor(labels).to(DEVICE),
            'scores': Tensor(scores).to(DEVICE)
        }
    else:
        return {
            'boxes': torch.stack(boxes, dim=0).to(DEVICE),
            'labels': IntTensor(labels).to(DEVICE),
            'scores': Tensor(scores).to(DEVICE)
        }

# 加载pkl对象
def loadObject(path):
    try:
        with open(path, 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except IOError:
        return None

def saveObject(obj, path):
    print("Saving " + path)
    with open(path, 'wb') as fid:
        pickle.dump(obj, fid)
    print(f"Saving finished {path}")

def filter_gt_from_all_pred(test_set, prediction_path, debug=True):
    pred_file_list = glob.glob(prediction_path + "*.pkl")
    image_id_map = {}
    for image in test_set["images"]:
        image_id_map[image["id"]] = image["file_name"]
    gt_dict = defaultdict(list)
    for item in test_set['annotations']:
        gt_dict[image_id_map[item["image_id"]]].append(item["category_id"])
    output = []
    # 解析预测数据，将label不在gt category中的那部分数据过滤
    for file in tqdm(pred_file_list):
        preds = loadObject(file)
        for pred in preds:
            result = {
                "boxes": [],
                "scores": [],
                "labels": [],
                "image_path": pred["img_path"]
            }
            image_name = pred["img_path"].split("/")[-1]
            cur_cate_ls = gt_dict[image_name]
            for i, label in enumerate(pred["labels"]):
                if label in cur_cate_ls:
                    result["boxes"].append(pred["boxes"][i])
                    result["scores"].append(pred["scores"][i])
                    result["labels"].append(pred["labels"][i])
            output.append(result)
    # 中间过程数据写入文件
    if debug:
        tmp_pkl_path = "./prediction_data2.pkl"
        saveObject(output, tmp_pkl_path)
    return output

def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path of the prediction that we want to evaluate')
    parser.add_argument('--debug', action="store_true",
                        help='Path of the prediction that we want to evaluate')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path of the ground truth path that we need')
    parser.add_argument('--out', type=str, default='results/results.txt', help='Where results will be stored')
    args = parser.parse_args()

    prediction_path = args.predictions
    ground_truth_path = args.ground_truth
    test_set = read_json(ground_truth_path)
    print("complete load test annotation data")
    # 将label不是gt类别的预测数据进行过滤
    pred_data = filter_gt_from_all_pred(test_set, prediction_path, args.debug)
    # 将ground truth数据按照image_ID维度进行group by，并将box格式进行转换
    ground_truth_data = get_all_image_ground_truth(test_set)
    # 将预测数据按照image_id维度进行group by
    preds_per_image = transform_predslist_to_dict(pred_data)
    # MAP
    iou_thresholds = [0.5] # 只指定一个iou阈值，降低map计算复杂度
    metric = MeanAveragePrecision(iou_thresholds = iou_thresholds).to(DEVICE)
    n_images = 0
    targets = []
    preds = []
    for imm in tqdm(test_set['images']):
        if imm['id'] in ground_truth_data:
            target = ground_truth_data[imm['id']]
            if imm['file_name'] in preds_per_image:
                pred = get_image_preds(preds_per_image[imm['file_name']])
            else:
                continue
            n_images += 1
            targets.append(target)
            preds.append(pred)
    metric.update(preds, targets)
    print("开始计算...")
    # Compute the results
    result = metric.compute()
    result['n_images'] = n_images
    result = {
        'map': float(result['map']),
        'map_50': float(result['map_50']),
        'map_75': float(result['map_75']),
        'map_small': float(result['map_small']),
        'map_medium': float(result['map_medium']),
        'map_large': float(result['map_large']),
        'mar_1': float(result['mar_1']),
        'mar_10': float(result['mar_10']),
        'mar_100': float(result['mar_100']),
        'mar_small': float(result['mar_small']),
        'mar_medium': float(result['mar_medium']),
        'mar_large': float(result['mar_large']),
        'map_per_class': float(result['map_per_class']),
        'mar_100_per_class': float(result['mar_100_per_class']),
        'n_images': int(result['n_images'])
    }

    print(result)

    with open("/data/chaihaojiang/our_data/GroundingDINO/result/result_demo.txt", "w") as json_file:
        json.dump(result, json_file)


if __name__ == '__main__':
    import sys
    test_args = ['evaluate_map.py', '--predictions', '/data/chaihaojiang/our_data/GroundingDINO/output/', '--ground_truth',
                 '/data/chaihaojiang/our_data/GroundingDINO/demo/new_annotations_test4.json', '--out', '/data/chaihaojiang/our_data/GroundingDINO/result'
                 ,'--debug']
    sys.argv = test_args  # Simulate command-line arguments
    main()




