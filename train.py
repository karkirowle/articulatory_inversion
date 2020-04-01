
import torch
import torch.nn as nn
from models import NeuralNet, DBLSTM, LinearRegression
import matplotlib.pyplot as plt

from nnmnkwii.datasets import PaddedFileSourceDataset
from data_utils import MFCCSource, ArticulatorySource, NanamiDataset, DummyDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

global_padded_length = 3500

mfcc_x = PaddedFileSourceDataset(MFCCSource("mngu0_wav/train"), padded_length=global_padded_length)
art_x = PaddedFileSourceDataset(ArticulatorySource("mngu0_ema/train"), padded_length=global_padded_length)
mfcc_x_test = PaddedFileSourceDataset(MFCCSource("mngu0_wav/eval"), padded_length=global_padded_length)
art_x_test = PaddedFileSourceDataset(ArticulatorySource("mngu0_ema/test"), padded_length=global_padded_length)

dataset = NanamiDataset(mfcc_x, art_x)
dataset_test = NanamiDataset(mfcc_x_test, art_x_test)

#dataset = DummyDataset(mfcc_x, art_x)
#dataset_test = DummyDataset(mfcc_x_test, art_x_test)

batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4)


test_loader= torch.utils.data.DataLoader(dataset_test,
                                             batch_size=1, shuffle=True,
                                             num_workers=4)


#pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths, batch_first=True)

input_size=20
#input_size = 1
hidden_size=300
hidden_size_2=100

#num_classes = 1
num_classes=12
num_epochs=50
learning_rate=1e-4


#model = NeuralNet(input_size, hidden_size, num_classes).to(device)
model = DBLSTM(input_size,batch_size,hidden_size,hidden_size_2,num_classes).to(device)
#model = LinearRegression(input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train = True
if train:
    total_step = len(train_loader)
    total_loss = 0
    for epoch in range(num_epochs):
        print("Epoch ", epoch + 1)
        for i, sample in enumerate(train_loader):
            # Convert numpy arrays to torch tensors
            inputs = sample['speech'].to(device)
            #inputs = inputs.o
            targets = sample['art'].to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss = total_loss / len(train_loader)
        print('Epoch [{}/{}], Train loss: {:.4f} cm'.format(epoch + 1, num_epochs, loss.item()))

        with torch.no_grad():
            total_loss = 0
            for i, sample in enumerate(test_loader):
                inputs = sample['speech'].to(device)
                targets = sample['art'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        loss = total_loss / len(test_loader)
        print('Epoch [{}/{}], Validation loss: {:.4f} cm'.format(epoch + 1, num_epochs, loss))
        torch.save(model.state_dict(), 'model_dblstm_' + str(epoch) + '.ckpt')
else:
    model.load_state_dict(torch.load("model_dblstm_20.ckpt"))
    #model.load_state_dict('model.ckpt')

    for i, sample in enumerate(test_loader):
        inputs = sample['speech'].to(device)
        targets = sample['art'].to(device)
        predicted = model(inputs).detach().cpu().numpy()
        targets=targets.detach().cpu().numpy()
        plt.plot(targets[0,:,0], label='Original data')
        plt.plot(predicted[0,:,0], label='Fitted line')
        plt.legend()
        plt.show()

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')