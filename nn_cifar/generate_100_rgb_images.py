
import cv2
import os
import torch
import pickle
import numpy as np

from nn_cifar.explore_images import apply, create_attention_filter

data_path = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/cifar-100-python"

X_train = np.zeros((50000, 3, 32, 32),dtype="uint8")
y_train = np.zeros((50000,),dtype="long")

with open("{0}/train".format(data_path), "rb") as file:
    _data = pickle.load(file, encoding='bytes')
    data = _data[b"data"]
    label = _data[b"fine_labels"]

    X_train = data.reshape(50000, 3, 32, 32)
    y_train = np.array(label).astype("long")

img_root = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/100/rgb_att_2"


# filter
# _filter = create_attention_filter(32,32,3,2)
_filter = None

for i,(img,label) in enumerate(zip(X_train,y_train)):
    img_category_root = "{0}/{1}".format(img_root,label)
    if not os.path.exists(img_category_root):
        os.mkdir(img_category_root)
    img = img.transpose((1,2,0))
    if _filter is not None:
        img = apply(img,_filter)

    cv2.imwrite("{0}/{1}.png".format(img_category_root,i),img)
