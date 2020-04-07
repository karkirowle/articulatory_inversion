import torch.nn as nn
import torch
import torch.nn.functional as F
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

class LinearRegression(nn.ModuleList):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, num_classes)


    def forward(self, x):
        out = self.fc1(x)
        return out


class DBLSTM(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, hidden_size_2, num_classes):
        super(DBLSTM, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        #self.dropout_1 = nn.Dropout(p=0.2)
        #self.dropout_2 = nn.Dropout(p=0.2)
        self.num_layers = 2
        self.hidden_dim = hidden_size
        self.lstm1 = nn.LSTM(hidden_size,hidden_size_2,bidirectional=True,num_layers=self.num_layers)
        self.fc3 = nn.Linear(hidden_size_2 * 2,num_classes)
        #self.batch_size = batch_size


    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))

        out, hidden = self.lstm1(out.view(out.shape[1],out.shape[0],-1))
        out = self.fc3(out.view(out.shape[1],out.shape[0],-1))
        return out


