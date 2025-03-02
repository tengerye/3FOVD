# 实验参考环境
Python: 3.9.19
CUDA: 11.8.89
PyTorch: 2.0.0+cu118
mmcv: 2.0.0
mmdet: 3.1.0

# 创建环境
conda create -n mmdetection python=3.9 -y
conda activate mmdetection

# 安装pytorch
conda install pytorch torchvision -c pytorch

# 安装mmcv
pip install -U openmim
mim install mmengine
mim install mmcv

# 进入项目文件夹
cd mmdetection
# 安装配置文件
pip install -v -e .

# 数据集准备
将coco格式数据集放入mmdetection/data/coco文件夹下

# 修改数据集对应类别名
vim/mmdet/datasets/coco.py
//修改METAINFO = { 'classes':( ) ,}部分中，类名为数据集对应类名

vim/mmdet/evaluation/functional/class_names.py
//修改def coco_classes() -> list:    return [ ]部分中，类名为数据集对应类名

# 修改各模型配置文件中对应类别数量，以centernet为例
vim/configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py
//修改num_classes为数据集对应类别数量

# 模型训练，以centernet为例
python tools/train.py configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py --work-dir WORKDIR


# 另，Co-Detr特殊，其模型在Co-DETR文件夹下，进入Co-DETR，前面安装过程一致，使用python tools/train.py /projects/co_deformable_detr/co_deformable_detr_r50_1x_coco.py --work-dir WORKDIR 进行训练

For training with multiple GPUs, we can use the bash `tools/dist_train.sh ${CONFIG_FILE}  ${GPU_NUM} --work-dir supervised/centernet/.

For test with multiple GPUs, we can use the bash `./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]`.