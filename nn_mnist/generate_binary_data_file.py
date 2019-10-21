import torch
from mlxtend.data import loadlocal_mnist



X, y = loadlocal_mnist(
        images_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/train-images-idx3-ubyte',
        labels_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/train-labels-idx1-ubyte')

Xt, yt = loadlocal_mnist(
        images_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/t10k-images-idx3-ubyte',
        labels_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/t10k-labels-idx1-ubyte')

data_dir = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_mnist/data"

def to_tensor(y,target):
    y = torch.from_numpy((y == target).astype('long'))
    torch.save(y,"{0}/{1}.pt".format(data_dir,"yt_" + str(target)))

for i in range(10):
    to_tensor(yt,i)


