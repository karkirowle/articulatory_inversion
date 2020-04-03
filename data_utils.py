


from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import scipy
import scipy.interpolate
from nnmnkwii.datasets import FileDataSource, FileSourceDataset, PaddedFileSourceDataset
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




        x, fs = librosa.load(wav_path,sr=16000)

        frame_time = 25 / 1000
        hop_time = 10 / 1000
        hop_length = int(hop_time * 16000)
        frame_length = int(frame_time * 16000)
        #print(frame_length)
        #print(fs)
        mfcc = librosa.feature.mfcc(x,sr=fs,hop_length=hop_length,n_mfcc=40,n_fft=frame_length).T


        return mfcc.astype(np.float32), wav_path


class ArticulatorySource(FileDataSource):
    def __init__(self,data_root,max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

    def collect_files(self):
        ema_paths = sorted(glob(join(self.data_root, "*.ema")))
        return ema_paths

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

    def collect_features(self, ema_path):
        #print("sajt")

        columns = {}
        columns["time"] = 0
        columns["present"] = 1

        with open(ema_path, 'rb') as f:

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

            data_out = data_[:, articulator_idx]

            if np.isnan(data_out).sum() != 0:
                # Build a cubic spline out of non-NaN values.
                spline = scipy.interpolate.splrep(np.argwhere(~np.isnan(data_out).ravel()),
                                                  data_out[~np.isnan(data_out)], k=3)
                # Interpolate missing values and replace them.
                for j in np.argwhere(np.isnan(data_out)).ravel():
                    data_out[j] = scipy.interpolate.splev(j, spline)

        return data_out, ema_path

class NanamiDataset(Dataset):
    """
    Generic wrapper around nnmnkwii datsets
    """
    def __init__(self,speech_padded_file_source,art_padded_file_source):
        self.speech = speech_padded_file_source
        self.art = art_padded_file_source

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        sample = {'speech': self.speech[idx][0], 'art': self.art[idx][0]}
        return sample

class InferenceDataset(Dataset):
    """
    Generic wrapper for nnmnkwii inference
    """
    def __init__(self,speech_padded_file_source):
        self.speech = speech_padded_file_source

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'speech': self.speech[idx][0], 'name': self.speech[idx][1]}
        return sample

class DummyDataset(Dataset):
    """
    Sinusoidal dummy dataset
    """
    def __init__(self,speech_padded_file_source,art_padded_file_source):
        self.speech = speech_padded_file_source
        self.art = art_padded_file_source

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        time = np.arange(0, 0.1, 0.0001)
        #print(idx, "Hz")
        f = idx
        speech = np.sin(2 * np.pi * f * time).astype(np.float32)
        speech = speech[:,None]

        art = 5 * np.cos(2 * np.pi * f * time).astype(np.float32)
        art = art[:,None]
        sample = {'speech': speech, 'art': art}
        return sample


def collate_wrapper(batch):

    max_duration = 0
    for idx in range(len(batch)):
        a = batch[idx]['speech'].shape[0]
        art = batch[idx]['art']
        art_temp = scipy.signal.resample(art, num=a)
        batch[idx]['art'] = art_temp
        if a > max_duration:
            max_duration = a

    speech = np.zeros((len(batch),max_duration,40))
    art = np.zeros((len(batch),max_duration,12))
    for idx in range(len(batch)):
        speech_temp = batch[idx]['speech']
        speech_duration = speech_temp.shape[0]
        art_temp = batch[idx]['art']
        art_duration = art_temp.shape[0]

        speech[idx,:speech_duration,:] = speech_temp
        art[idx,:art_duration,:] = art_temp

    sample = {'speech': torch.FloatTensor(speech), 'art': torch.FloatTensor(art)}
    return sample


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    # First commit is capable of




    mfcc_x = FileSourceDataset(MFCCSource("mngu0_wav/test"))

    mfcc, name = mfcc_x[2]

    #print(mfcc)
    print(name)

    #plt.imshow(mfcc_x[2],aspect="auto")
    #plt.show()
    art_x = FileSourceDataset(ArticulatorySource("mngu0_ema/test"))



    dataset = NanamiDataset(mfcc_x, art_x)

    for i in range(len(dataset)):
        dataset[i]


    art, name = art_x[2]
    print(name)

