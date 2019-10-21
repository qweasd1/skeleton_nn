from torch import nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self,output=2):
        super(CNNModel, self).__init__()

        # all

        ## experiment -1 training time: 787s, train-acc 0.999, test-acc 0.9907
        self.channel_1 = 8
        self.channel_2 = 16
        self.hidden_1 = 50

        # self.channel_1 = 20
        # self.channel_2 = 30
        # self.hidden_1 = 100


        # binary

        ## experiment - 1, training time: 369s, train-acc 0.999, test-acc 0.9855
        # self.channel_1 = 8
        # self.channel_2 = 16
        # self.hidden_1 = 50

        self.output = output
        self.conv1 = nn.Conv2d(1, self.channel_1, 5, 1)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * self.channel_2, self.hidden_1)
        self.fc2 = nn.Linear(self.hidden_1, self.output)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * self.channel_2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def num_features(self,x):
        shape = x.size()[1:]
        result = 1
        for size in shape:
            result *= size
        return result

    def calc_accuracy(self,train_data):
        correct_count = 0
        for X,y in train_data.batches():
            correct_count += self(X).argmax(axis=1).eq(y).sum().item()
        return correct_count / train_data.size