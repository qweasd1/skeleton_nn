import torch
from mlxtend.data import loadlocal_mnist



# X, y = loadlocal_mnist(
#         images_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/train-images-idx3-ubyte',
#         labels_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/train-labels-idx1-ubyte')

Xt, yt = loadlocal_mnist(
        images_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/t10k-images-idx3-ubyte',
        labels_path='/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/files/MNIST/raw/t10k-labels-idx1-ubyte')

data_dir = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_mnist/data"

def to_tensor(X,y,X_filename, y_filename):
    X_norm = (X / 256 - 0.1307) / 0.3081
    X_norm = torch.from_numpy(X_norm.reshape((-1,1,28,28)).astype('float32'))
    y = torch.from_numpy(y.astype('long'))
    torch.save(X_norm,"{0}/{1}.pt".format(data_dir,X_filename))
    torch.save(y,"{0}/{1}.pt".format(data_dir,y_filename))


# to_tensor(X,y,"X","y")
to_tensor(Xt,yt,"Xt","yt")


