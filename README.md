# Articulatory inversion

This is a DBLSTM baseline model based on the work of

- (1) [LIU, Peng, et al. A deep recurrent approach for acoustic-to-articulatory inversion. In: 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015. p. 4450-4454.](https://ieeexplore.ieee.org/abstract/document/7178812)
- (2) [Zhu, Pengcheng / Xie, Lei / Chen, Yunlin (2015): "Articulatory movement prediction using deep bidirectional long short-term memory based recurrent neural networks and word/phone embeddings", In INTERSPEECH-2015, 2192-2196.](https://www.isca-speech.org/archive/interspeech_2015/papers/i15_2192.pdf)

The repository of [articulatory inversion from bootphon](https://github.com/bootphon/articulatory_inversion/blob/master/Training/train.py)
was also a great help in keeping my sanity. A few code lines were shamelessly copied.

### Quick start

After cloning and cd-ing into the repository 

```
python3 -m venv venv 
sourve venv/bin/activate
pip install -r requirements.txt
```

If you don't like virtualenv, list of most important packages
- nnmnkwii 0.0.20
- numpy 1.18.2
- torch 1.4.0
- matplotlib 3.2.1
- sklearn 0.22.2.post1
- scipy 1.4.1
- ConfigArgParse 1.2
- Tensorboard 2.2.0

### Preprocessing

The files can run with and without preprocessing. It is recommended to preprocess,
because recalculating the FFT slows the training.

If you preprocess run first,
```
python3 preprocessing.py
```
Otherwise, rewrite MFCCSourceNPY to MFCCSource in train.py (flag will be provided in later version)

### Inference
After downloading the [model](https://drive.google.com/drive/folders/1DY7uF2HuW-oUpUmjjvuuNbkpZXrBAYrv?usp=sharing) and putting everything in the right directory
the model should work with the default:
```
python3 infer.py
```
You can consult the help of the inference file for more info on non-default arguments.
```
python3 infer.py --help
```

### Train
```
python3 train.py --train --my-conf=configs/BLSTM.conf
```



### Modifications from paper (1)

I sticked to more standard LSTM tuning procedures, which explains the difference
in test results partly. However, rigorous comparison with the paper is impossible as
they haven't provided the train-test partitioning.

I assume that the measurement data is in cm (not given in dataset, but mentioned in [MNGU0 forum](http://www.mngu0.org/messages/problems-bugs-etc/195679262))

- ADAM optimisation instead of RMSProp
- librosa MFCC (40) instead of STRAIGHT (?) LSF (40)
- ReLU is used on the FCN layers, instead of tanh. 
- No context windows. LSTM should be able to model dynamic information.
- We sticked to the full retraining, instead of the layerwise pretraining
- 50 epochs best validation loss instead of early stopping
- PyTorch default initialisaion are used
### Modifications from paper (2)

Because MFCC is available in librosa I decided to make a more faithful
implementation of the BLSTM. There are things that are still not evident from the
paper, so I will contact the authors about these.

- Batch size is not mentioned in the paper, so we used 64
- Lower incisor is mentioned in the paper instead of jaw. That seems to be consistent
across MNGU0 papers, though
- No regularisation is mentioned in the paper
- It is not clear whether they used layerwise pretraining for the model

| Model | Papers result | Our result |
| ----- | ------------- | ---------- |
| BLSTM (1) | 0.963 mm | 1.077 mm | 
| BLSTM (2) | 0.565 mm | 1.833 mm | 

### Implementational nuances

- RMSE is defined in a slightly complicated way, we divide by the sequence length
and sum for each sample. The last batch has less samples so it is reweighted to get a more
faithful estimation of RMSE.
- Number of mel filterbanks is set to 40 from librosa default 128. Otherwise we would
get empty responses.

### Contributions

Contributions are welcome. The general desire with this repository is
to keep the preprocessing code as minimal as possible (no intervention to the data). Thus,
contributions should focus on better neural network models. (seq2seq)

