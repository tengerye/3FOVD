import copy
import multiprocessing
import os
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datetime import datetime

# 全局缓存区（通过fork机制共享）
_shared_cache = {}

def get_nowtime():
    now = datetime.now()
    formatted_time = now.strftime('%Y/%m/%d %H:%M:%S')
    return formatted_time

def _init_worker(shared_cache):
    """
    子进程初始化验证（确保数据存在）
    """
    global _shared_cache
    _shared_cache = shared_cache
    assert _shared_cache['coco_gt'] is not None, "数据未正确共享"


def evaluate_chunk(args):
    """
    子进程评估函数，处理指定类别的评估任务
    """
    iou_type, chunk_cat_ids, params = args
    # 每个子进程独立加载数据（注意内存消耗）
    pid = os.getpid()
    coco_gt = _shared_cache['coco_gt']
    coco_dt = _shared_cache['coco_dt']
    # 初始化评估器
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    # 继承主进程参数设置
    coco_eval.params = params
    # 设置当前进程需要处理的类别ID
    coco_eval.params.catIds = chunk_cat_ids
    # 执行评估
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
    global _shared_cache
    _shared_cache['coco_gt'] = COCO(gt_json_path)
    _shared_cache['coco_dt'] = _shared_cache['coco_gt'].loadRes(pred_json_path)
    print(f"{get_nowtime()} 数据加载完毕")
    # 初始化主评估器获取参数
    main_evaluator = COCOeval(_shared_cache['coco_gt'], _shared_cache['coco_dt'], iouType=iou_type)
    params = copy.deepcopy(main_evaluator.params)

    # 获取所有类别ID并分块
    all_cat_ids = _shared_cache['coco_gt'].getCatIds()
    chunks = np.array_split(all_cat_ids, num_processes)

    # 准备多进程参数
    task_args = [
        (iou_type, chunk.tolist(), params)
        for chunk in chunks
    ]

    # 创建进程池
    with multiprocessing.Pool(processes=num_processes, initializer=_init_worker,  # 初始化worker验证数据
        initargs=(_shared_cache,)) as pool:
        results = pool.map(evaluate_chunk, task_args)

    # 合并所有子进程结果
    # main_evaluator.evalImgs = [
    #     img for chunk_results in results for img in chunk_results
    # ]
    # 上面的合并子进程返回值的方式太低效，这里采用更高效的合并方式
    total_len = sum(len(chunk) for chunk in results)
    print(f"{get_nowtime()} 数组开始预分配")
    merged = np.empty(total_len, dtype=object)
    print(f"{get_nowtime()} 数组结束分配")
    offset = 0
    for chunk in results:
        merged[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
    print(f"{get_nowtime()} 子进程数据合并结束")
    main_evaluator.evalImgs = merged.tolist()
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
        "/root/post_process_data/dino/vehicle/instances_vehicle_test.json",
        "/root/post_process_data/dino/vehicle/groundingdino_vehicle_test_prediction_remove_cover_v2.json",
        num_processes=10
    )
    # metrics = parallel_coco_evaluation(
    #     "/data/data/final/product/product_yolo/valid/annotations/instances_product_valid.json",
    #     "/data/chaihaojiang/postprocess_data_0305/vild/vild_product_val_prediction_visualized_coco_baseline.json",
    #     num_processes=2
    # )
    end_time = time.time()
    print(f"共耗时{end_time - start_time}秒")
    print(metrics)

