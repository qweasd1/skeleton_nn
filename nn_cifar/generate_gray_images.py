
import cv2
import os
import torch
import pickle
import numpy as np
data_path = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/cifar-10-batches-py"

X_train = np.zeros((50000, 3, 32, 32),dtype="uint8")
y_train = np.zeros((50000,),dtype="long")


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

if __name__ == '__main__':

    for i in range(5):
        with open("{0}/data_batch_{1}".format(data_path,i+1),"rb") as file:
            _data = pickle.load(file,encoding='bytes')
            data = _data[b"data"]
            label = _data[b"labels"]

            X_train[i * 10000:(i + 1) * 10000] = data.reshape(10000, 3, 32, 32)
            y_train[i * 10000:(i + 1) * 10000] = np.array(label).astype("long")


    img_root = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/rgb"
    img_gray_root = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/gray"
    img_edge_root = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/edge"

    for i,(img,label) in enumerate(zip(X_train,y_train)):
        img_category_root = "{0}/{1}".format(img_root,label)
        if not os.path.exists(img_category_root):
            os.mkdir(img_category_root)
        img = img.transpose((1,2,0))
        cv2.imwrite("{0}/{1}.png".format(img_category_root,i),img)

        img_category_root = "{0}/{1}".format(img_gray_root, label)
        if not os.path.exists(img_category_root):
            os.mkdir(img_category_root)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("{0}/{1}.png".format(img_category_root, i), img_gray)

        img_category_root = "{0}/{1}".format(img_edge_root, label)
        if not os.path.exists(img_category_root):
            os.mkdir(img_category_root)
        img_edge = auto_canny(img_gray)
        cv2.imwrite("{0}/{1}.png".format(img_category_root, i), img_edge)