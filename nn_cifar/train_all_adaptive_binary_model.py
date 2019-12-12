from nn_cifar.load_train_data import AdaptiveTrainData, AddModelException
from nn_cifar.cnn_model import CNNModel
import torch
from torch.optim import Adam
import torch.nn.functional as F
import time

models_root = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/better_models"
for target in range(10):
    X_filepath = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/X_att2.pt"
    y_filepath = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/y_{0}.pt".format(target)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = AdaptiveTrainData(X_filepath,y_filepath,device)


    model = CNNModel(2)

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    print("=====start training: model {1} with parameters: {0} ===".format(get_n_params(model),target))

    model.to(device)
    optimizer = Adam(model.parameters(),lr=0.001)


    start = time.time()
    acc_change_limit = 3


    prev_acc = -1
    acc_duplicate_count = 0
    while True:
        iteration_start = time.time()
        for i, (X, y) in enumerate(train_data.batches()):
            optimizer.zero_grad()
            out = model(X)
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
        iteration_end = time.time()
        try:
            acc = train_data.evaluate_and_expand(model)
        except AddModelException:
            print("new model created")
            model = CNNModel(2).to(device)
            optimizer = Adam(model.parameters(), lr=0.001)
            acc_duplicate_count = 0
            prev_acc = -1
            if not train_data.new_model():

                break
        # if acc > 0.99:
        #     optimizer = Adam(model.parameters(), lr=0.0001)

        if acc != prev_acc:
            acc_duplicate_count = 0
        else:
            acc_duplicate_count += 1
            if acc_duplicate_count == acc_change_limit:
                model = CNNModel(2).to(device)
                acc_duplicate_count = 0
                prev_acc = -1
        print(acc_duplicate_count)
        prev_acc = acc
        message = "{0} (total {1}| current {2})".format(acc,iteration_end - start, iteration_end - iteration_start)
        print(message)

    train_data.save_model(models_root,target)




