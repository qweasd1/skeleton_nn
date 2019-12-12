import torch
import pickle
import numpy as np
import cv2

from nn_cifar.generate_gray_images import auto_canny

data_path = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/cifar-100-python"

X_train = np.zeros((50000, 3, 32, 32),dtype="uint8")
y_train = np.zeros((50000,),dtype="long")


with open("{0}/train".format(data_path),"rb") as file:
        _data = pickle.load(file,encoding='bytes')
        data = _data[b"data"]
        label = _data[b'fine_labels']

        X_train = data.reshape(50000, 3, 32, 32)
        y_train = np.array(label).astype("long")

def normalize_img(matrix):
    matrix = matrix.astype("float32")
    means = np.average(matrix, axis=(0, 2, 3))
    vars = np.var(matrix, axis=(0, 2, 3))

    for i in range(matrix.shape[1]):
        matrix[:, i, :, :] = (X_train[:, i, :, :] - means[i]) / vars[i]
    return matrix


# X, y = loadlocal_mnist(
#         images_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/train-images-idx3-ubyte',
#         labels_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/train-labels-idx1-ubyte')
#
# Xt, yt = loadlocal_mnist(
#         images_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/train-images-idx3-ubyte',
#         labels_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/train-labels-idx1-ubyte')
#
data_dir = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/100"
#
def to_tensor(X,X_filename):
    X_norm = torch.from_numpy(X)
    torch.save(X_norm,"{0}/{1}.pt".format(data_dir,X_filename))

# generate original file
to_tensor(normalize_img(X_train),"X")
to_tensor(y_train,"y")

# generate gray training file
# X_train_gray = np.zeros((50000,1,32,32))
# X_train_edge = np.zeros((50000,1,32,32))
#
# for i,img in enumerate(X_train):
#     img_gray = cv2.cvtColor(img.transpose((1, 2, 0)), cv2.COLOR_BGR2GRAY)
#     img_edge = auto_canny(img_gray)
#     X_train_gray[i,0] = img_gray
#     X_train_edge[i,0] = img_edge
#
# to_tensor(normalize_img(X_train_gray),"X_gray")
# to_tensor(normalize_img(X_train_edge),"X_edge")
# to_tensor(X,y,"Xt","yt")


