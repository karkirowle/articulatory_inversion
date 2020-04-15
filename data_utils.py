import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import scipy
import scipy.interpolate
from torch.nn.utils.rnn import pad_sequence
from nnmnkwii.datasets import FileDataSource, FileSourceDataset, PaddedFileSourceDataset
import librosa
import itertools


class MFCCSource(FileDataSource):
    def __init__(self,data_root,max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

    def collect_files(self):
        files = open(self.data_root).read().splitlines()

        # Because of a,b,c,d,e,f not being included, we need a more brute force globbing approach here
        files_wav = list(map(lambda x: "mngu0_wav/all/" + x + "*.wav", files))
        all_files = [glob(files) for files in files_wav]
        all_files_flattened = list(itertools.chain(*all_files))
        return all_files_flattened

    def collect_features(self, wav_path):
        x, fs = librosa.load(wav_path,sr=16000)
        frame_time = 10 / 1000
        hop_time = 5 / 1000
        hop_length = int(hop_time * 16000)
        frame_length = int(frame_time * 16000)
        n_mels = 40
        mfcc = librosa.feature.mfcc(x,sr=fs,hop_length=hop_length,n_mfcc=40,n_fft=frame_length,n_mels=n_mels).T

        return mfcc.astype(np.float32), wav_path


class SpectrogramSource(MFCCSource):
    def __init__(self,data_root,max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

    def collect_features(self, wav_path):
        x, fs = librosa.load(wav_path,sr=16000)
        frame_time = 10 / 1000
        hop_time = 5 / 1000
        hop_length = int(hop_time * 16000)
        frame_length = int(frame_time * 16000)

        melspectrogram = librosa.feature.melspectrogram(x,sr=fs,hop_length=hop_length,n_fft=frame_length).T

        return melspectrogram.astype(np.float32), wav_path


class MFCCSourceNPY(FileDataSource):
    def __init__(self,data_root,max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

    def collect_files(self):
        files = open(self.data_root).read().splitlines()
        # Because of a,b,c,d,e,f not being included, we need a more brute force globbing approach here
        files_wav = list(map(lambda x: "mngu0_wav/npy/" + x + "*.wav.npy", files))
        all_files = [glob(files) for files in files_wav]
        all_files_flattened = list(itertools.chain(*all_files))
        return all_files_flattened

    def collect_features(self, npy_path):

        mfcc = np.load(npy_path)
        return mfcc.astype(np.float32), npy_path

class SpectogramSourceNPY(FileDataSource):
    def __init__(self,data_root,max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

    def collect_files(self):
        files = open(self.data_root).read().splitlines()

        # Because of a,b,c,d,e,f not being included, we need a more brute force globbing approach here
        files_wav = list(map(lambda x: "mngu0_wav/spect_npy/" + x + "*.wav.npy", files))
        all_files = [glob(files) for files in files_wav]
        all_files_flattened = list(itertools.chain(*all_files))
        return all_files_flattened

    def collect_features(self, npy_path):
        spect = np.load(npy_path)
        return spect, npy_path


class PPGSource(FileDataSource):
    def __init__(self,data_root,max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

    def collect_files(self):
        files = open(self.data_root).read().splitlines()
        # Because of a,b,c,d,e,f not being included, we need a more brute force globbing approach here
        files_wav = list(map(lambda x: "mngu0_ppg/" + x + "*.npy", files))
        all_files = [glob(files) for files in files_wav]
        all_files_flattened = list(itertools.chain(*all_files))
        return all_files_flattened

    def collect_features(self, wav_path):

        x = np.load(wav_path)
        return x, wav_path


class ArticulatorySource(FileDataSource):
    def __init__(self,data_root,max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

    def collect_files(self):
        files = open(self.data_root).read().splitlines()
        files_wav = list(map(lambda x: "mngu0_ema/all/" + x + "*.ema", files))
        all_files = [glob(files) for files in files_wav]
        all_files_flattened = list(itertools.chain(*all_files))
        return all_files_flattened

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
                'T1_py', 'T1_px', 'T3_py', 'T3_px', 'T2_py', 'T2_px',
                'jaw_py', 'jaw_px', 'upperlip_py', 'upperlip_px',
                'lowerlip_py', 'lowerlip_px']
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

        art_temp = scipy.signal.resample(self.art[idx][0], num=self.speech[idx][0].shape[0])
        return (torch.FloatTensor(self.speech[idx][0]), torch.FloatTensor(art_temp))


class NanamiDataset2(FileDataSource):
    """
    Generic wrapper around nnmnkwii datsets
    """
    def __init__(self,speech_padded_file_source,spect_padded_file_source,art_padded_file_source):
        self.speech = speech_padded_file_source
        self.art = art_padded_file_source
        self.spect = spect_padded_file_source

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        art_temp = scipy.signal.resample(self.art[idx][0], num=self.speech[idx][0].shape[0])
        return (torch.FloatTensor(self.speech[idx][0]),
                torch.FloatTensor(art_temp),
                torch.FloatTensor(self.spect[idx][0]))

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

        #return self.speech[idx][0], self.speech[idx][1]
        sample = {'speech': self.speech[idx][0], 'name': self.speech[idx][1]}
        return sample


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    mask = pad_sequence([torch.ones_like(y) for y in yy], batch_first=True, padding_value=0)
    return xx_pad,  yy_pad, x_lens, y_lens, mask

def pad_collate_2(batch):
    (xx, yy,zz) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    z_lens = [len(z) for z in zz]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    zz_pad = pad_sequence(zz, batch_first=True, padding_value=0)


    mask = pad_sequence([torch.ones_like(y) for y in yy], batch_first=True, padding_value=0)
    return xx_pad,  yy_pad, zz_pad, x_lens, y_lens, z_lens, mask


