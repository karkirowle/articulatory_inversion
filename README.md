# Articulatory inversion

This is a DBLSTM baseline model based on the work of

[LIU, Peng, et al. A deep recurrent approach for acoustic-to-articulatory inversion. In: 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015. p. 4450-4454.](https://ieeexplore.ieee.org/abstract/document/7178812)

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
python3 train.py --train
```


### Modifications from paper

I sticked to more standard LSTM tuning procedures, which explains the difference
in test results partly. However, rigorous comparison with the paper is impossible as
they haven't provided the train-test partitioning.

I assume that the measurement data is in cm (not given in dataset, but mentioned in [MNGU0 forum](http://www.mngu0.org/messages/problems-bugs-etc/195679262))

- ADAM optimisation instead of RMSProp
- librosa MFCC (40) instead of LSF (40)
- ReLU is used on the FCN layers, instead of tanh. 
- No context windows. LSTM should be able to model dynamic information.


### Notes on why is it difficult to compare baselines with each other

- Some works smooth/low-pass filter EMA signal (?)
- No train-test partition, and even if there is, no guarantee that architectures
are not optimised on test set (bad practice)
- Lot of preprocessing trickery is employed, not just simple switch
of frontends

### Contributions

Contributions are welcome. The general desire with this repository is
to keep the preprocessing code as minimal as possible (no intervention to the data). Thus,
contributions should focus on better neural network models. (seq2seq)

### Train-test partitioning

See [train_test.txt here](https://drive.google.com/drive/folders/1DY7uF2HuW-oUpUmjjvuuNbkpZXrBAYrv?usp=sharing).