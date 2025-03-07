import json
import random

from tqdm import tqdm


def sample_dataset(input_path, output_path, sample_size=1000):
    # 读取原始数据
    with open(input_path, 'r') as f:
        data = json.load(f)

    # 获取所有唯一的image_id
    all_image_ids = set()
    for item in tqdm(data):
        all_image_ids.add(item["image_id"])
        if len(all_image_ids) >= 10000:
            break

    # 随机抽取指定数量的image_id
    sampled_image_ids = random.sample(list(all_image_ids), sample_size)

    # 过滤出选中image_id对应的数据
    sampled_data = []
    for item in tqdm(data):
        if item['image_id'] in sampled_image_ids:
            sampled_data.append(item)

    # 保存新数据集
    with open(output_path, 'w') as f:
        json.dump(sampled_data, f)

    print(f"已成功抽取 {len(sampled_data)} 条数据，包含 {sample_size} 张图片")


if __name__ == '__main__':
    input_path = "/root/post_process_data/vild/vehicle/vild_car_test_prediction_remove_cover_v2.json"
    output_path = "/root/post_process_data/vild/vehicle/vild_car_test_prediction_remove_cover_v2_sample.json"
    sample_dataset(input_path, output_path)
    pass





















