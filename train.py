
import torch
import torch.nn as nn
from models import DBLSTM
from attention_model import AttentionGRU
import matplotlib.pyplot as plt
import numpy as np
from nnmnkwii.datasets import FileSourceDataset
from data_utils import MFCCSource, ArticulatorySource, NanamiDataset, collate_wrapper
import argparse

# Device configuration


def train(train):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    mfcc_x = FileSourceDataset(MFCCSource("mngu0_wav/train"))
    art_x = FileSourceDataset(ArticulatorySource("mngu0_ema/train"))

    mfcc_x_test = FileSourceDataset(MFCCSource("mngu0_wav/test"))
    art_x_test = FileSourceDataset(ArticulatorySource("mngu0_ema/test"))

    dataset = NanamiDataset(mfcc_x, art_x)
    dataset_test = NanamiDataset(mfcc_x_test, art_x_test)



    batch_size = 2

    train_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=4, collate_fn=collate_wrapper)

    test_loader= torch.utils.data.DataLoader(dataset_test,
                                                 batch_size=1, shuffle=True,
                                                 num_workers=4,collate_fn=collate_wrapper)



    # DBLSTM Parameters
    input_size=40
    hidden_size=300
    hidden_size_2=100

    num_classes=12
    num_epochs=50
    learning_rate=1e-4

    #model = DBLSTM(input_size,batch_size,hidden_size,hidden_size_2,num_classes).to(device)

    # AttentionGRU models
    emb_dim = 128
    enc_hid_dim = 128
    dec_hid_dim = 128
    dropout = 0.0
    model = AttentionGRU(input_size,emb_dim,enc_hid_dim,dec_hid_dim,num_classes,dropout).to(device)


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
                print(i)
                # Convert numpy arrays to torch tensors
                inputs = sample['speech'].to(device)
                #inputs = inputs.o
                targets = sample['art'].to(device)
                # Forward pass
                #outputs = model(inputs)
                outputs = model(inputs,targets)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss = np.sqrt(total_loss / len(train_loader))
            print('Epoch [{}/{}], Train RMSE: {:.4f} cm'.format(epoch + 1, num_epochs, loss.item()))

            with torch.no_grad():
                total_loss = 0
                for i, sample in enumerate(test_loader):
                    inputs = sample['speech'].to(device)
                    targets = sample['art'].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()

            loss = np.sqrt(total_loss / len(test_loader))
            print('Epoch [{}/{}], Test RMSE: {:.4f} cm'.format(epoch + 1, num_epochs, loss))
            torch.save(model.state_dict(), 'model_attention_' + str(epoch) + '.ckpt')
    else:
        model.load_state_dict(torch.load("model_dblstm_48.ckpt"))
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


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train and save a model.')

    parser.add_argument('--train', type=bool, default= False,
                        help='whether to save one graph of prediction & target of the test ')

    args = parser.parse_args()

    train(args.train)