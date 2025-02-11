from ultralytics import YOLO
 
 
model = YOLO("D:\\ultralytics4.12\\runs\\detect\\train16\\weights\\best.pt")
model.predict(source="datasets/images/val",save=True,save_conf=True,save_txt=True,name='output')
 
#source后为要预测的图片数据集的的路径
#save=True为保存预测结果
#save_conf=True为保存坐标信息
#save_txt=True为保存txt结果，但是yolov8本身当图片中预测不到异物时，不产生txt文件