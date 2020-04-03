
import torch
import matplotlib.pyplot as plt
from models import DBLSTM
from nnmnkwii.datasets import FileSourceDataset
from data_utils import MFCCSource, InferenceDataset
import numpy as np
from os.path import split, join
import argparse

def infer(model_name,wav_dir,save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = 40
    hidden_size = 300
    hidden_size_2 = 100
    num_classes = 12
    batch_size = 1
    model = DBLSTM(input_size, batch_size, hidden_size, hidden_size_2, num_classes).to(device)

    mfcc_x_test = FileSourceDataset(MFCCSource(wav_dir))
    dataset_test = InferenceDataset(mfcc_x_test)
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=1, shuffle=False,
                                              num_workers=4)

    model.load_state_dict(torch.load(model_name))

    for i, sample in enumerate(test_loader):
        inputs = sample['speech'].to(device)
        wav, filename = mfcc_x_test[i]

        filename_save = join(save_path,split(filename)[1].split(".")[0])
        predicted = model(inputs).detach().cpu().numpy()

        plt.plot(predicted[0,:,:])
        plt.show()
        np.save(filename_save,predicted[0,:,:])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and save a model.')

    parser.add_argument('--model', default="model_dblstm_48.ckpt",
                        help='PyTorch ckpt to use')
    parser.add_argument('--wav_dir', default="mngu0_wav/test",
                        help='Directory with wav files to do the articulatory inversion on')
    parser.add_argument('--save_dir', default="/home/boomkin/repos/articulatory_inversion/mngu0_ema/predicted",
                        help='Directory to save articulatory inversion results to (.npy extension all)')
    args = parser.parse_args()


    infer(args.model,args.wav_dir,args.save_dir)

