import torch.nn as nn
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
    def __init__(self, args):
        super(DBLSTM, self).__init__()
        self.fc1 = nn.Linear(args.input_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size,args.hidden_size)
        self.num_layers = 2
        self.hidden_dim = args.hidden_size
        self.lstm1 = nn.LSTM(args.hidden_size,args.hidden_size_2,bidirectional=True,num_layers=self.num_layers)
        self.fc3 = nn.Linear(args.hidden_size_2 * 2,args.num_classes)


    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))

        out, hidden = self.lstm1(out.view(out.shape[1],out.shape[0],-1))
        out = self.fc3(out.view(out.shape[1],out.shape[0],-1))
        return out


