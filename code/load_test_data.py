import os
import json
import csv
import pandas as pd

# 类别和caption的对应字典
class_caption_dict = {}
# 英文类别名称和中文类别名称的对应关系
ch2en_dict = {}

base_path = "/home/username/"
ground_truth_img_path = base_path + "image/"
ground_truth_label_path = base_path + "label/"
det_parent_path = "shop_det_data"
gt_parent_path = "shop_gt_data"
model_path = "./result/map/model.names"
# 测试case
test_data = {}


def init_data():
    csv_file_path = f"{base_path}序号_en.csv"
    class_trans_excel_path = f"{base_path}class_trans.xlsx"

    data_frame = pd.read_excel(class_trans_excel_path, sheet_name='Sheet2')
    # 填充en2ch_dict数据
    for i in range(data_frame.__len__()):
        ch_classname = data_frame.at[i, 'ch']
        ch_classname = ch_classname.strip()
        en_classname = data_frame.at[i, 'en']
        ch2en_dict[ch_classname] = en_classname

    # 填充 class_caption_dict 数据
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num != 1:
                class_name = row[1]
                content = row[11]
                if '/' in class_name:
                    classname_ls = class_name.split('/')
                    for name in classname_ls:
                        caption = name + "#" + content
                        class_caption_dict[name] = caption
                else:
                    caption = class_name + "#" + content
                    class_caption_dict[class_name] = caption


class ImageObject:
    def __init__(self, img_id, img_path, json_path, box_info):
        self.img_id = img_id
        self.img_path = img_path
        self.json_path = json_path
        self.box_info = box_info
        self.class_en_list = []
        self.caption_dict = {}  # {class: caption}

    def fill_caption_list(self):
        # 两条正样本数据填充
        for box in self.box_info:
            classname = box['label']
            if ch2en_dict.get(classname) is not None:
                classname_en = ch2en_dict[classname]
                if class_caption_dict.get(classname_en) is not None:
                    self.class_en_list.append(classname_en)
                    c = class_caption_dict[classname_en]
                    self.caption_dict[classname] = c

    def __str__(self):
        return (f"ImageObject(img_id={self.img_id}, img_path={self.img_path}, json_path={self.json_path}, "
                f"box_info={self.box_info}), class_en_list={self.class_en_list}, caption_dict={self.caption_dict}")


def init_img_obj():
    # 图片对象
    image_obj_list = []
    # 存放照片的列表
    img_list = []
    img_id_list = []
    for file in os.listdir(ground_truth_img_path):
        img_id = file.split(".")[0]
        filename = os.path.join(ground_truth_img_path, file)
        if filename.endswith(".jpg"):
            img_list.append(filename)
            img_id_list.append(img_id)

    for img_id in img_id_list:
        img_path = ground_truth_img_path + str(img_id) + ".jpg"
        json_path = ground_truth_label_path + str(img_id) + ".json"
        with open(json_path, "r") as f:
            box_info = json.load(f)['shapes']
        image_obj = ImageObject(img_id, img_path, json_path, box_info)
        image_obj.fill_caption_list()
        image_obj_list.append(image_obj)
        print(f".....img_id: {img_id}处理完成.....")
    return image_obj_list


def init():
    init_data()
    image_obj_list = init_img_obj()
    for img in image_obj_list:
        # img_id = img.img_id
        # img_path = ground_truth_img_path + img_id + ".jpg"

        test_data[img.img_path] = list(class_caption_dict.values())
    class_list = list(class_caption_dict.keys())
    # 写model_name文件
    with open(model_path, "a+") as f:
        for class_name in class_list:
            f.write(class_name + "\n")
    for img_obj in image_obj_list:
        boxes = img_obj.box_info
        filename = img_obj.json_path.split("/")[-1].split(".")[0] + ".txt"
        path = f"./result/map/{gt_parent_path}/" + filename
        img_path = f"./result/map/{det_parent_path}/" + filename
        for data in boxes:
            class_name = data["label"]
            if class_name in list(ch2en_dict.keys()):
                class_name_en = ch2en_dict[class_name]
                if class_name_en in class_list:
                    class_ = class_list.index(class_name_en)
                    x1 = data["points"][0][0]
                    y1 = data["points"][0][1]
                    x2 = data["points"][1][0]
                    y2 = data["points"][1][1]
                    with open(path, 'a+') as f:
                        f.write(f"{class_} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
                    open(img_path, 'w').close()

    return test_data, image_obj_list, class_caption_dict


def init_data2():
    csv_file_path = f"{base_path}/valid/shop_en_no.csv"

    # 填充 class_caption_dict 数据
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num != 1:
                class_name = row[1]
                content = row[2]
                caption = class_name + "#" + content
                class_caption_dict[class_name] = caption


if __name__ == '__main__':
    # init()
    init2()
