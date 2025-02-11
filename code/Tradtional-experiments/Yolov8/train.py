from ultralytics import YOLO
import multiprocessing
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch

    # Train the model

    results = model.train(data="D:\\ultralytics4.12\\ultralytics\\cfg\\datasets\\coco128.yaml", epochs=300, imgsz=(1333, 800),device='cpu')


if __name__ == '__main__':
    multiprocessing.freeze_support()

    main()
