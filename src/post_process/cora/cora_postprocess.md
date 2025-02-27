

1.  Follow the [official install instruction](https://github.com/tgxs002/CORA) to install the environment.

2. 将此cora文件夹下的文件除了readme.md外全部移动到官方仓库的根目录下

3. 运行start.sh脚本

   start.sh脚本运行demo:

   > sh start.sh ./annotations.json ./images/ ./caption_csv_path/ car baseline  ./checkpoints/COCO_RN50.pth ./checkpoints/region_prompt_R50.pth

    参数解释释（按照传入位置顺序）

	1. annotations.json: 标注文件    2.  ./images/： 图片存放路径   3.  caption_csv_path: caption描述文件路径   

	4. 如果运行的是车辆数据集传入car；如果是商品数据集传入product    5. 如果是基线实验，传入baseline; 如果是imp实验, 传入imp2

	6和7分别传入cora的权重参数路径