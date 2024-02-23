# 把dataset中的数据集划分为训练集和测试集，比例为8:2，保存到当前目录下的train和val文件夹中

import os
import shutil
import random
import torch
import torch.utils.data


# 读取文件夹下的所有文件名
def read_file_name(file_dir):
    files_list = []
    if not os.path.exists(file_dir):
        print("文件夹不存在！")
        return
    for root, dirs, files in os.walk(file_dir):
        if len(files) != 0:
            files_list.extend(
                [os.path.join(root, file) for file in files]
            )  # 将文件名添加到列表中
    return files_list


# 划分数据集
def split_dataset(data_dir, train_ratio=0.8):
    # 读取文件夹下的所有文件名
    images = read_file_name(os.path.join(data_dir, "images/images"))
    labels = read_file_name(os.path.join(data_dir, "label/label"))
    # 打乱files
    print("数据集总样本数：", len(images))
    print("标签总样本数：", len(labels))

    # 计算训练集和测试集的样本数量
    # total_samples = len(images)
    # train_samples = int(total_samples * train_ratio)
    # val_samples = total_samples - train_samples

    # 创建保存训练集和测试集的文件夹
    train_dir = "./train"
    val_dir = "./val"
    dirs = [train_dir, val_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
        else:
            shutil.rmtree(dir)
            os.mkdir(dir)

    # 划分数据集
    train_files, val_files = torch.utils.data.random_split(
        images, [train_ratio, 1-train_ratio]
    )
    # 正则化字符串得到样本名字

    # 将文件复制到训练集和测试集文件夹中
    for file in train_files:
        shutil.copy(file, train_dir)
        file_name = file.split("/")[-1]
        # print(os.path.join(data_dir, "label/label") + file_name[:-4] + ".xml")
        # if os.path.join(data_dir, "label/label", file_name[:-4] + ".xml") in labels:
        convert_to_yolo(
            os.path.join(data_dir, "label/label", file_name[:-4] + ".xml"),
            os.path.join(train_dir, file_name[:-4]) + ".txt",
        )

    for file in val_files:
        shutil.copy(file, val_dir)
        file_name = file.split("/")[-1]
        # if os.path.join(data_dir, "label/label", file_name[:-4] + ".xml") in labels:
        convert_to_yolo(
            os.path.join(data_dir, "label/label", file_name[:-4] + ".xml"),
            os.path.join(val_dir, file_name[:-4]) + ".txt",
        )


import xml.etree.ElementTree as ET


def parse_xml(xml_path):
    """解析 XML 标签文件。

    Args:
        xml_path: XML 标签文件路径。

    Returns:
        解析后的 XML 标签对象。
    """
    if not os.path.exists(xml_path):
        print(xml_path, " XML 标签文件不存在！")
        return
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return root


def convert_to_yolo(xml_path, yolo_path):
    """将 XML 标签转化为 YOLO 格式标签。

    Args:
        xml_path: XML 标签文件路径。
        yolo_path: YOLO 格式标签文件路径。
    """
    if not os.path.exists(xml_path):
        with open(yolo_path, "a") as f:
            pass
        return
    root = parse_xml(xml_path)
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)
    for obj in root.findall("object"):
        try:
            class_name = int((obj.find("name").text).split("_")[0]) - 1
        except:
            print((obj.find("name").text).split("_")[0])
            print(xml_path)
            assert(0)
        x_min = int(obj.find("bndbox").find("xmin").text)
        y_min = int(obj.find("bndbox").find("ymin").text)
        x_max = int(obj.find("bndbox").find("xmax").text)
        y_max = int(obj.find("bndbox").find("ymax").text)
        x = (x_min + x_max) / 2 / width
        y = (y_min + y_max) / 2 / height
        w = (x_max - x_min) / width
        h = (y_max - y_min) / height

        with open(yolo_path, "a") as f:
            f.write(
                "{} {} {} {} {}\n".format(
                    class_name, round(x, 6), round(y, 6), round(w, 6), round(h, 6)
                )
            )


if __name__ == "__main__":
    # label_dir = "../dataset/Defects location for metal surface/label/label"
    # images_dir = "../dataset/Defects location for metal surface/images/images"
    data_dir = "../dataset/Defects location for metal surface"
    split_dataset(data_dir)

    # get train.txt and validation.txt
    txt = ["./train.txt", "./val.txt"]
    dir_list = ["./train", "./val"]
    if os.path.exists(txt[0]):
        os.remove(txt[0])
    if os.path.exists(txt[1]):
        os.remove(txt[1])

    for i in range(2):
        list_files = read_file_name(dir_list[i])
        list_files = sorted(list_files)
        for file in list_files:
            if file[-3:] == "jpg":
                with open(txt[i], "a") as f:
                    # 把文件绝对路径写入txt
                    f.write(os.path.abspath(file) + "\n")
    pass
