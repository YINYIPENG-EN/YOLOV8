import os
import random
import shutil
from tqdm import tqdm


def split_datasets(datasets_path, data_dir, dataset_img_list, desc=''):
    '''
    :param datasets_path: 数据集根目录
    :param data_dir: train or val 根目录，eg:datasets/target/train/ or datasets/target/val/
    :param dataset_img_list:train or val images files list
    :param desc:tqdm of str train or val
    '''
    for img_file in tqdm(dataset_img_list, desc=desc):
        # 检查是否为图像
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            source_img = os.path.join(images_path, img_file)  # 源图像路径
            Dataset_path = os.path.join(data_dir, 'images', img_file)  # 目标训练集路径
            # 复制文件到训练到train_dataset
            shutil.copy(source_img, Dataset_path)
            # 操作对应label
            # 获取label name
            label_file_name = img_file.split('.')[0] + '.txt'
            source_label_path = os.path.join(datasets_path, 'labels')  # 获取label文件路径
            source_label_file_path = os.path.join(source_label_path, label_file_name) # 获取对应label.txt路径
            Label_path = os.path.join(data_dir, 'labels', label_file_name) # label目标文件路径
            # 复制Label文件到train_label
            shutil.copy(source_label_file_path, Label_path)


datasets_path = 'cfg/datasets/target'  # root path
images_path = os.path.join(datasets_path, 'images')
train_dir = os.path.join(datasets_path, 'train') # train_root
val_dir = os.path.join(datasets_path, 'val')  # val root
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(train_dir + '/images', exist_ok=True)
os.makedirs(val_dir + '/images', exist_ok=True)
os.makedirs(train_dir + '/labels',exist_ok=True)
os.makedirs(val_dir + '/labels',exist_ok=True)
images_file_list = os.listdir(images_path)  # 获取所有的图像文件，xxx.jpg
random.shuffle(images_file_list)  # 打乱
# 划分train和val两个数据集
num_train = int(len(images_file_list) * 0.9)
train_img_files_list = images_file_list[:num_train]
val_img_files_list = images_file_list[num_train:]
# 划分train
split_datasets(datasets_path,train_dir,train_img_files_list)
# 划分val
split_datasets(datasets_path, val_dir, val_img_files_list)




