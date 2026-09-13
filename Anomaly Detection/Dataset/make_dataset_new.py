

import numpy as np 
import os 
import shutil 
import cv2
from pycocotools.coco import COCO
import glob

from datasets200.VisA import Visa_dataset
from datasets200.MVTec import Mvtec_dataset




def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)

def copy_images(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        return

    os.makedirs(dst_dir, exist_ok=True)

    for file_name in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, file_name)

        if not os.path.isfile(src_path):
            continue

        dst_path = os.path.join(dst_dir, file_name)
        shutil.copy2(src_path, dst_path)


def make_loco_dataset(src_root, dst_root):
    categories = [
        "breakfast_box",
        "juice_bottle",
        "pushpins",
        "screw_bag",
        "splicing_connectors",
    ]

    move(dst_root)

    for category in categories:
        print("Processing :", category)

        category_src = os.path.join(src_root, category)
        category_dst = os.path.join(dst_root, category)

        # 训练正常样本
        copy_images(
            os.path.join(category_src, "train", "good"),
            os.path.join(category_dst, "train", "good")
        )

        # 测试正常样本
        copy_images(
            os.path.join(category_src, "test", "good"),
            os.path.join(category_dst, "test", "good")
        )

        # 测试逻辑异常和结构异常
        for anomaly_type in [
            "logical_anomalies",
            "structural_anomalies",
        ]:
            copy_images(
                os.path.join(category_src, "test", anomaly_type),
                os.path.join(category_dst, "test", anomaly_type)
            )

            # ground truth mask
            copy_images(
                os.path.join(category_src, "ground_truth", anomaly_type),
                os.path.join(category_dst, "ground_truth", anomaly_type)
            )

    print("MVTec LOCO finished !")


    

if __name__ == "__main__":
    
    #des_root = "./dataset/mvisa/data/visa"   # generated unified dataset path
    #move(des_root)
    #id = 0

    #Visa_Dataset = Visa_dataset("Your root path/visa")   # original dataset path
    #id = Visa_Dataset.make_VAND(binary=True,to_255=True,des_path_root=des_root,id=id) 
    #print(id)


    #-------------------------------------------------------------------------------
    

    
    # #des_root = "./dataset/mvisa/data/mvtec"   # generated unified dataset path
    # move(des_root)
    # id = 0

    # Mvtec_Dataset = Mvtec_dataset(r"E:\A\VCP-CLIP\dataset\mvtec_anomaly_detection")  #original dataset path
    # id = Mvtec_Dataset.make_VAND(binary=True,to_255=True,des_path_root=des_root,id=id)
    # print(id)

    src_root = (
            "/data_16t/longjinhao/caozihao/"
            "VCP-CLIP-new/dataset/mvtec_loco_anomaly_detection"
        )
    
    dst_root = (
            "/data_16t/longjinhao/caozihao/"
            "VCP-CLIP-new/dataset/mvisa/data/mvtec_loco"
        )
    
    make_loco_dataset(src_root, dst_root)

    


    
