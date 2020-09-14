import os
import numpy as np
import cv2
import sys
import glob
import pickle
from multiprocessing import Pool
import pandas as pd

label_dic = np.load('label_dir.npy', allow_pickle=True).item()
print(label_dic)


def convert_csv_to_dict(csv_path):
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.iloc[i, :]
        slash_rows = data.iloc[i, 0].split('/')
        class_name = slash_rows[0]
        basename = slash_rows[1].split('.')[0]

        keys.append(basename)
        key_labels.append(class_name)

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        label = key_labels[i]
        database[key] =label

    return database

source_dir = 'data/UCF-101'
target_train_dir = 'data/splits/train'
target_test_dir = 'data/splits/test'
dir_path = 'ucfTrainTestlist/'
split_index=1
train_csv_path = dir_path + 'trainlist0{}.txt'.format(split_index)
val_csv_path = dir_path + 'testlist0{}.txt'.format(split_index)


if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)


traindatabase = convert_csv_to_dict(train_csv_path)
testdatabase = convert_csv_to_dict(val_csv_path)

for key in label_dic:
    each_mulu = key + '_jpg'
    print(each_mulu, key)

    label_dir = os.path.join(source_dir, each_mulu)
    label_mulu = os.listdir(label_dir)

    for each_label_mulu in label_mulu:
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu))
        image_file.sort()
        image_name = image_file[0][:-6]
        image_num = len(image_file)
        frame = []
        vid = image_name
        for i in range(image_num):
            image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_name + '_' + str(i+1) + '.jpg')
            frame.append(image_path)

        output_pkl = vid + '.pkl'
        if vid in traindatabase.keys():
            output_pkl = os.path.join(target_train_dir, output_pkl)
        elif vid in testdatabase.keys():
            output_pkl = os.path.join(target_test_dir, output_pkl)


        f = open(output_pkl, 'wb')
        pickle.dump((vid, label_dic[key], frame), f, -1)
        f.close()
