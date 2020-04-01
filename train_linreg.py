
import torch
import torch.nn as nn
from models import NeuralNet, DBLSTM
import matplotlib.pyplot as plt

from nnmnkwii.datasets import PaddedFileSourceDataset
from data_utils import MFCCSource, ArticulatorySource, NanamiDataset, DummyDataset
from sklearn.linear_model import LinearRegression

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global_padded_length = 3500

mfcc_x = PaddedFileSourceDataset(MFCCSource("mngu0_wav/train"), padded_length=global_padded_length)
art_x = PaddedFileSourceDataset(ArticulatorySource("mngu0_ema/train"), padded_length=global_padded_length)
mfcc_x_test = PaddedFileSourceDataset(MFCCSource("mngu0_wav/eval"), padded_length=global_padded_length)
art_x_test = PaddedFileSourceDataset(ArticulatorySource("mngu0_ema/test"), padded_length=global_padded_length)

reg = LinearRegression().fit(X, y)