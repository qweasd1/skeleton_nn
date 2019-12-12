import torch
import pickle
import numpy as np
data_path = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/cifar-10-batches-py"


y_train = np.zeros((10000,),dtype="long")


for i in range(1):
    with open("{0}/test_batch".format(data_path),"rb") as file:
        _data = pickle.load(file,encoding='bytes')
        label = _data[b"labels"]
        y_train[i * 10000:(i + 1) * 10000] = np.array(label).astype("long")


data_dir = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data"

def to_tensor(y,target):
    y = torch.from_numpy((y == target).astype('long'))
    torch.save(y,"{0}/{1}.pt".format(data_dir,"yt_" + str(target)))

for i in range(10):
    to_tensor(y_train,i)


