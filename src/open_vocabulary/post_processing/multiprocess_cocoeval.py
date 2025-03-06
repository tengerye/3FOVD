import copy
import multiprocessing
import os
import time

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datetime import datetime

def get_nowtime():
    now = datetime.now()
    formatted_time = now.strftime('%Y/%m/%d %H:%M:%S')
    return formatted_time

def evaluate_chunk(args):
    """
    子进程评估函数，处理指定类别的评估任务
    """
    gt_json_path, pred_json_path, iou_type, chunk_cat_ids, params = args
    # 每个子进程独立加载数据（注意内存消耗）
    pid = os.getpid()
    print(f"{get_nowtime()} 进程 {pid} 开始加载数据")
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)
    print(f"{get_nowtime()} 进程 {pid} 结束加载数据")
    # 初始化评估器
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    # 继承主进程参数设置
    coco_eval.params = params
    # 设置当前进程需要处理的类别ID
    coco_eval.params.catIds = chunk_cat_ids
    # 执行评估
    pid = os.getpid()
    print(f"{get_nowtime()} 进程 {pid} 开始评估")
    coco_eval.evaluate()
    print(f"{get_nowtime()} 进程 {pid} 结束评估")
    return coco_eval.evalImgs  # 返回评估结果


def parallel_coco_evaluation(gt_json_path, pred_json_path, iou_type='bbox', num_processes=4):
    """
    并行化COCO评估主函数
    """
    # 主进程加载基础数据
    print(f"{get_nowtime()} 加载数据")
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)
    print(f"{get_nowtime()} 数据加载完毕")
    # 初始化主评估器获取参数
    main_evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    params = copy.deepcopy(main_evaluator.params)

    # 获取所有类别ID并分块
    all_cat_ids = coco_gt.getCatIds()
    chunks = np.array_split(all_cat_ids, num_processes)

    # 准备多进程参数
    task_args = [
        (gt_json_path, pred_json_path, iou_type, chunk.tolist(), params)
        for chunk in chunks
    ]

    # 创建进程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(evaluate_chunk, task_args)

    # 合并所有子进程结果
    main_evaluator.evalImgs = [
        img for chunk_results in results for img in chunk_results
    ]
    print(f"{get_nowtime()} 所有进程evaluate结束")
    # 后续聚合计算
    main_evaluator._paramsEval = copy.deepcopy(main_evaluator.params)
    main_evaluator.accumulate()
    main_evaluator.summarize()

    # 返回标准格式结果
    return {
        "AP": main_evaluator.stats[0],
        "AP50": main_evaluator.stats[1],
        "AP75": main_evaluator.stats[2],
        "AP_small": main_evaluator.stats[3],
        "AP_medium": main_evaluator.stats[4],
        "AP_large": main_evaluator.stats[5],
        "AR1": main_evaluator.stats[6],
        "AR10": main_evaluator.stats[7],
        "AR100": main_evaluator.stats[8],
        "AR_small": main_evaluator.stats[9],
        "AR_medium": main_evaluator.stats[10],
        "AR_large": main_evaluator.stats[11],
    }


if __name__ == '__main__':
    # 使用示例
    start_time = time.time()
    metrics = parallel_coco_evaluation(
        "/root/post_process_data/vild/product/instances_product_test.json",
        "/root/post_process_data/dino/groundingdino_product_test_prediction_remove_cover_v2.json",
        num_processes=10
    )
    end_time = time.time()
    print(f"共耗时{end_time - start_time}秒")
    print(metrics)

