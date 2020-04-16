
import torch
import random
from models import Modern_DBLSTM_1, DBLSTM_LayerNorm, Traditional_BLSTM_2, Modern_BLSTM_2
import matplotlib.pyplot as plt
import numpy as np
from nnmnkwii.datasets import FileSourceDataset
from data_utils import ArticulatorySource, NanamiDataset, pad_collate, MFCCSource, MFCCSourceNPY, LSFSource, NormalisedArticulatorySource
import configargparse
from configs import configs
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping

def worker_init_fn(worker_id):
    # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(manual_seed + worker_id)

def train(args):

    print(args.__dict__)
    writer = SummaryWriter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.MFCC:
        # Load MNGU0 features
        train_x = FileSourceDataset(MFCCSourceNPY("trainfiles.txt"))
        val_x = FileSourceDataset(MFCCSourceNPY("validationfiles.txt"))
        test_x = FileSourceDataset(MFCCSourceNPY("testfiles.txt"))
    elif args.LSF:
        # Load LSF features
        train_x = FileSourceDataset(LSFSource("trainfiles.txt"))
        val_x = FileSourceDataset(LSFSource("validationfiles.txt"))
        test_x = FileSourceDataset(LSFSource("testfiles.txt"))
    else:
        raise NameError("No frontend loaded!")

    if args.art_norm:

        train_y = FileSourceDataset(NormalisedArticulatorySource("trainfiles.txt"))
        val_y = FileSourceDataset(NormalisedArticulatorySource("validationfiles.txt"))
        test_y = FileSourceDataset(NormalisedArticulatorySource("testfiles.txt"))

        # Loads normalisaion means and standard deviations for metric handling - 4 times because z-score norm
        mngu0_mean = np.genfromtxt("mngu0_ema/all_normalised/norm_parms/ema_means.txt")
        ema_mean = torch.FloatTensor(mngu0_mean[:12]).to(device)
        mngu0_std = np.genfromtxt("mngu0_ema/all_normalised/norm_parms/ema_stds.txt")
        ema_std = torch.FloatTensor(mngu0_std[:12]).to(device)

    else:
        train_y = FileSourceDataset(ArticulatorySource("trainfiles.txt"))
        val_y = FileSourceDataset(ArticulatorySource("validationfiles.txt"))
        test_y = FileSourceDataset(ArticulatorySource("testfiles.txt"))

    dataset = NanamiDataset(train_x, train_y)
    dataset_val = NanamiDataset(val_x, val_y)
    dataset_test = NanamiDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size, shuffle=True,
                                                 num_workers=4, collate_fn=pad_collate)

    val_loader = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=args.batch_size, shuffle=True,
                                              num_workers=4, collate_fn=pad_collate)

    test_loader = torch.utils.data.DataLoader(dataset_test,
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=4,collate_fn=pad_collate)

    if args.Traditional_BLSTM_2:
        model = Traditional_BLSTM_2(args).to(device)
    if args.Modern_BLSTM_2:
        model = Modern_BLSTM_2(args).to(device)
    if args.Modern_BLSTM_1:
        model = Modern_DBLSTM_1(args).to(device)

    if args.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
    if args.Adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    es = EarlyStopping()

    # Train the model

    epoch_log = np.zeros((args.num_epochs,3))

    if args.train:
        writer.add_hparams(args.__dict__,{'started':True})

        for epoch in range(args.num_epochs):
            total_loss = 0
            print("Epoch ", epoch + 1)
            for i, sample in enumerate(train_loader):
                xx_pad,  yy_pad, _, _, mask = sample

                inputs = xx_pad.to(device)

                targets = yy_pad.to(device)
                if args.art_norm:
                    targets = (targets + ema_mean)*(4 * ema_std)


                mask = mask.to(device)

                outputs = model(inputs)
                if args.art_norm:
                    outputs = (outputs + ema_mean)*(4 * ema_std)


                loss = torch.sum(((outputs - targets) * mask) ** 2.0) / torch.sum(mask).item()

                if targets.shape[0] == args.batch_size:
                    total_loss += loss.item()
                else:
                    total_loss += loss.item() * (targets.shape[0] / args.batch_size)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss = np.sqrt(total_loss / len(train_loader))
            writer.add_scalar('Loss/Train', total_loss, epoch + 1)
            epoch_log[epoch,0] = total_loss
            print('Epoch [{}/{}], Train RMSE: {:.4f} cm'.format(epoch + 1, args.num_epochs, total_loss))

            torch.cuda.empty_cache()

            with torch.no_grad():
                total_loss = 0
                for i, sample in enumerate(val_loader):
                    xx_pad, yy_pad, _, _, mask = sample
                    inputs = xx_pad.to(device)
                    targets = yy_pad.to(device)
                    if args.art_norm:
                        targets = (targets + ema_mean) * (4 * ema_std)
                    mask = mask.to(device)

                    outputs = model(inputs)

                    if args.art_norm:
                        outputs = (outputs + ema_mean) * (4 * ema_std)

                    loss = torch.sum(((outputs-targets)*mask)**2.0) / torch.sum(mask).item()

                    # Weigh differently the last smaller batch
                    if targets.shape[0] == args.batch_size:
                        total_loss += loss.item()
                    else:
                        total_loss += loss.item() * (targets.shape[0]/args.batch_size)

                total_loss = np.sqrt(total_loss / len(val_loader))
                writer.add_scalar('Loss/Validation', total_loss, epoch + 1)
                epoch_log[epoch, 1] = total_loss
                print('Epoch [{}/{}], Validation RMSE: {:.4f} cm'.format(epoch + 1, args.num_epochs, total_loss))

            torch.cuda.empty_cache()

            with torch.no_grad():
                total_loss = 0
                for i, sample in enumerate(test_loader):
                    xx_pad, yy_pad, _, _, mask = sample
                    inputs = xx_pad.to(device)
                    targets = yy_pad.to(device)
                    if args.art_norm:
                        targets = (targets + ema_mean) * (4 * ema_std)
                    mask = mask.to(device)

                    outputs = model(inputs)
                    if args.art_norm:
                        outputs = (outputs + ema_mean) * (4 * ema_std)
                    loss = torch.sum(((outputs-targets)*mask)**2.0) / torch.sum(mask).item()
                    if targets.shape[0] == args.batch_size:
                        total_loss += loss.item()
                    else:
                        total_loss += loss.item() * (targets.shape[0] / args.batch_size)

                total_loss = np.sqrt(total_loss / len(test_loader))
                writer.add_scalar('Loss/Test', total_loss, epoch + 1)
                epoch_log[epoch, 2] = total_loss
                print('Epoch [{}/{}], Test RMSE: {:.4f} cm'.format(epoch + 1, args.num_epochs, total_loss))

            torch.cuda.empty_cache()

            if es.call(epoch_log[epoch,1]):
                best_id = np.argmin(epoch_log[:i, 1])
                print("Early stopping! Best test loss: ", epoch_log[best_id,2])
                break
        best_id = np.argmin(epoch_log[:i,1])
        writer.add_hparams(args.__dict__,
                      {'hparam/train': epoch_log[best_id,0], 'hparam/val': epoch_log[best_id,1], 'hparam/test': epoch_log[best_id,2]})
        writer.close()
    else:
        model.load_state_dict(torch.load("model_dblstm_48.ckpt"))

        for i, sample in enumerate(test_loader):
            xx_pad, yy_pad, _, _, _ = sample
            inputs = xx_pad.to(device)
            targets = yy_pad.to(device)

            predicted = model(inputs).detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            plt.plot(targets[0,:,0], label='Original data')
            plt.plot(predicted[0,:,0], label='Fitted line')
            plt.legend()
            plt.show()


if __name__ == '__main__':

    p = configargparse.ArgParser()
    p.add('-c', '--my-config', required=True, is_config_file=True, help='config file path')

    args = configs.parse(p)

    manual_seed = 2
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    train(args)