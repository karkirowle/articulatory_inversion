
import torch
import torch.nn as nn
from models import DBLSTM
from attention_model import AttentionGRU
import matplotlib.pyplot as plt
import numpy as np
from nnmnkwii.datasets import FileSourceDataset
from data_utils import MFCCSource, ArticulatorySource, NanamiDataset, pad_collate
import configargparse
from configs import configs

def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mfcc_x = FileSourceDataset(MFCCSource("trainfiles.txt"))
    art_x = FileSourceDataset(ArticulatorySource("trainfiles.txt"))

    mfcc_x_test = FileSourceDataset(MFCCSource("testfiles.txt"))
    art_x_test = FileSourceDataset(ArticulatorySource("testfiles.txt"))

    dataset = NanamiDataset(mfcc_x, art_x)
    dataset_test = NanamiDataset(mfcc_x_test, art_x_test)

    train_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size, shuffle=True,
                                                 num_workers=4, collate_fn=pad_collate)

    test_loader = torch.utils.data.DataLoader(dataset_test,
                                                 batch_size=1, shuffle=True,
                                                 num_workers=4,collate_fn=pad_collate)


    if args.BLSTM:
        model = DBLSTM(args).to(device)
    if args.attention:
        model = AttentionGRU(args).to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model

    if args.train:
        total_loss = 0
        for epoch in range(args.num_epochs):
            print("Epoch ", epoch + 1)
            for i, sample in enumerate(train_loader):
                xx_pad,  yy_pad, _, _, mask = sample
                # Convert numpy arrays to torch tensors
                inputs = xx_pad.to(device)
                targets = yy_pad.to(device)
                mask = mask.to(device)
                # Forward pass

                if args.BLSTM:
                    outputs = model(inputs)
                if args.attention:
                    outputs = model(inputs,targets)

                loss = torch.sum(((outputs - targets) * mask) ** 2.0) / torch.sum(mask)
                total_loss += loss.item()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss = np.sqrt(total_loss / len(train_loader))
            print('Epoch [{}/{}], Train RMSE: {:.4f} cm'.format(epoch + 1, args.num_epochs, total_loss))

            with torch.no_grad():
                total_loss = 0
                for i, sample in enumerate(test_loader):
                    xx_pad, yy_pad, _, _, mask = sample
                    inputs = xx_pad.to(device)
                    targets = yy_pad.to(device)
                    mask = mask.to(device)
                    if args.BLSTM:
                        outputs = model(inputs)
                    if args.attention:
                        outputs = model(inputs, targets)
                    loss = torch.sum(((outputs-targets)*mask)**2.0) / torch.sum(mask)
                    total_loss += loss.item()

                total_loss = np.sqrt(total_loss / len(test_loader))
                print('Epoch [{}/{}], Test RMSE: {:.4f} cm'.format(epoch + 1, args.num_epochs, total_loss))
                torch.save(model.state_dict(), 'model_dblstm_bs_2' + str(epoch) + '.ckpt')
    else:
        model.load_state_dict(torch.load("model_dblstm_48.ckpt"))

        for i, sample in enumerate(test_loader):
            xx_pad, yy_pad, _, _, _ = sample
            inputs = xx_pad.to(device)
            targets = yy_pad.to(device)
            if args.BLSTM:
                outputs = model(inputs)
            if args.attention:
                outputs = model(inputs, targets)

            predicted = model(inputs).detach().cpu().numpy()
            targets=targets.detach().cpu().numpy()
            plt.plot(targets[0,:,0], label='Original data')
            plt.plot(predicted[0,:,0], label='Fitted line')
            plt.legend()
            plt.show()


if __name__ == '__main__':

    p = configargparse.ArgParser()
    p.add('-c', '--my-config', required=True, is_config_file=True, help='config file path')

    args = configs.parse(p)

    train(args)