


import os
from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import scipy
import scipy.interpolate
from nnmnkwii.datasets import FileDataSource, FileSourceDataset
import librosa


class MFCCSource(FileDataSource):
    def __init__(self,data_root,max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

    def collect_files(self):

        wav_paths = sorted(glob(join(self.data_root, "*.wav")))
        label_paths = wav_paths
        if self.max_files is not None and self.max_files > 0:
            return wav_paths[:self.max_files], label_paths[:self.max_files]
        else:
            return wav_paths, label_paths

    def collect_features(self, wav_path, label_path):
        x, fs = librosa.load(wav_path)
        mfcc = librosa.feature.mfcc(x)

        return mfcc.astype(np.float32)


class ArticulatoryDataset(Dataset):
    """Articulatory dataset."""

    def __init__(self, root_dir_input, pad=3408, transform=None):
        """
        Args:
            root_dir_input (string): Directory with all the EMA
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir_input = root_dir_input

        # Fetching the glob to calculate the number of samples
        self.all_ema_files = glob(os.path.join(root_dir_input,"*.ema"))
        self.sample_num = len(self.all_ema_files)
        self.pad = pad
        self.transform = transform

    def __len__(self):
        return self.sample_num

    def clean(self,s):
        """
        Strips the new line character from the buffer input
        Parameters:
        -----------
        s: Byte buffer
        Returns:
        --------
        p: string stripped from new-line character
        """
        s = str(s, "utf-8")
        return s.rstrip('\n').strip()

    def __ema__(self, path: str) -> np.float32:
        """
        Reads in a single EMA file

        parameters:
        -----------
        path: Filename with extension .ema (String)
        bias: Implicit bias in shape (it is needed in one of the datasets)

        returns:
        --------
        data: 2D numpy array with sample points x channel
        """

        columns = {}
        columns["time"] = 0
        columns["present"] = 1

        with open(path, 'rb') as f:

            dummy_line = f.readline()  # EST File Track
            datatype = self.clean(f.readline()).split()[1]
            nframes = int(self.clean(f.readline()).split()[1])
            f.readline()  # Byte Order
            nchannels = int(self.clean(f.readline()).split()[1])

            while not 'CommentChar' in str(f.readline(), "utf-8"):
                pass
            f.readline()  # empty line
            line = self.clean(f.readline())

            while not "EST_Header_End" in line:
                channel_number = int(line.split()[0].split('_')[1]) + 2
                channel_name = line.split()[1]
                columns[channel_name] = channel_number
                line = self.clean(f.readline())

            string = f.read()
            data = np.fromstring(string, dtype='float32')
            data_ = np.reshape(data, (-1, len(columns)))

            # There is a list of columns here we can select from, but looking around github, mostly the below are used

            articulators = [
                'T1_py', 'T1_pz', 'T3_py', 'T3_pz', 'T2_py', 'T2_pz',
                'jaw_py', 'jaw_pz', 'upperlip_py', 'upperlip_pz',
                'lowerlip_py', 'lowerlip_pz']
            articulator_idx = [columns[articulator] for articulator in articulators]

            data_out = data_[:,articulator_idx]


            if np.isnan(data_out).sum() != 0:
                # Build a cubic spline out of non-NaN values.
                spline = scipy.interpolate.splrep( np.argwhere(~np.isnan(data_out).ravel()), data_out[~np.isnan(data_out)], k=3)
                # Interpolate missing values and replace them.
                for j in np.argwhere(np.isnan(data_out)).ravel():
                    data_out[j] = scipy.interpolate.splev(j, spline)

            return data_out

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = os.path.join(self.all_ema_files[idx])
        articulation = self.__ema__(path).astype(np.float64)

        return articulation




if __name__ == '__main__':
    print("sajt")

    import matplotlib.pyplot as plt
    # First commit is capable of

    mfcc_x = FileSourceDataset(MFCCSource("mngu0_wav/train"))

    # Iterating through the data extracting MFCCs
    for i in range(len(mfcc_x)):
       plt.imshow(mfcc_x[i],aspect="auto")
       plt.show()


    #Iterating through the EMA showing plots

    audio_x = ArticulatoryDataset(root_dir_input="mngu0_ema/train", root_dir_output="")

    for i in range(len(audio_x)):
        plt.plot(audio_x[i])
        plt.show()