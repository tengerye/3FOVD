import os
import cv2
import mmcv
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'D:\\mmdetection\\configs\\faster_rcnn\\faster-rcnn_r50-caffe-c4_ms-1x_coco.py'
checkpoint_file = 'D:\\mmdetection\\fastrcnnFGVD\\epoch_5.pth'

model = init_detector(config_file, checkpoint_file, device='cpu')

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta


def detect_image(model, visualizer, img_path):
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)

    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        show=False)
    img_with_bbox = visualizer.get_image()

    save_path = img_path.replace('.jpg', '_result.jpg')
    cv2.imwrite(save_path, img_with_bbox[:, :, ::-1])


def detect_images_in_folder(model, visualizer, folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
    for img_path in image_paths:
        detect_image(model, visualizer, img_path)


def main():
    # 图片预测
    folder_path = 'images/'
    detect_images_in_folder(model, visualizer, folder_path)


if __name__ == '__main__':
    main()