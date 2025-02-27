1. Follow the [official install instruction](https://github.com/IDEA-Research/GroundingDINO) to install the environment.
2. 用当前./util/inference.py 替换掉 官方groundingdino/util/inference.py文件
3. 将此groundingdino文件夹下的文件除了readme.md外全部移动到官方仓库的根目录下
4. 运行start.sh脚本

```
> sh start.sh ./annotations.json ./images/ ./caption_csv_path/ car baseline 
```

参数解释释（按照传入位置顺序）

1. annotations.json: 标注文件    2.  ./images/： 图片存放路径   3.  caption_csv_path: caption描述文件路径   

2. 如果运行的是车辆数据集传入car；如果是商品数据集传入product    5. 如果是基线实验，传入baseline; 如果是imp实验, 传入imp2