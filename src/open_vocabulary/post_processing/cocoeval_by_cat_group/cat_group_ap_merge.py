import json
import glob

if __name__ == '__main__':
    res_path = "/root/exp/aps/group_*_ap.json"
    output_path = "/root/exp/result"
    # 收集所有结果文件
    result_files = glob.glob(res_path)

    # 合并AP结果
    all_aps = {}
    for fpath in result_files:
        with open(fpath, 'r') as f:
            aps = json.load(f)
            all_aps.update(aps)

    # 计算mAP
    mAP = sum(all_aps.values()) / len(all_aps)
    print(f'Final mAP: {mAP:.4f}')

    # 可选：保存详细结果
    with open(f'{output_path}/final_aps.json', 'w') as f:
        json.dump({
            'per_category_ap': all_aps,
            'mAP': mAP
        }, f, indent=2)