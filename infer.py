
import torch
import matplotlib.pyplot as plt
from models import DBLSTM
from nnmnkwii.datasets import FileSourceDataset
from data_utils import MFCCSource, InferenceDataset
import numpy as np
from os.path import split, join

import configargparse
from configs import configs
def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = DBLSTM(args).to(device)

    mfcc_x_test = FileSourceDataset(MFCCSource(args.wav_dir))
    dataset_test = InferenceDataset(mfcc_x_test)
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=1, shuffle=False,
                                              num_workers=4)

    model.load_state_dict(torch.load(args.model_name))

    for i, sample in enumerate(test_loader):
        inputs = sample['speech'].to(device)

        wav, filename = mfcc_x_test[i]

        filename_save = join(args.save_dir,split(filename)[1].split(".")[0])
        predicted = model(inputs).detach().cpu().numpy()

        plt.plot(predicted[0,:,:])
        plt.show()
        np.save(filename_save,predicted[0,:,:])

if __name__ == '__main__':

    parser = configargparse.ArgParser()

    parser.add_argument('--model_name', default="rmse_0_1077.ckpt",
                        help='PyTorch ckpt to use')
    parser.add_argument('--wav_dir', default="testfiles.txt",
                        help='Directory with wav files to do the articulatory inversion on')
    parser.add_argument('--save_dir', default="/home/boomkin/repos/articulatory_inversion/mngu0_ema/predicted",
                        help='Directory to save articulatory inversion results to (.npy extension all)')

    args = configs.parse(parser)


    infer(args)

