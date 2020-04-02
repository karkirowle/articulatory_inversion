

from glob import glob
import numpy as np
from shutil import copyfile
from os.path import split

path = "/home/boomkin/Downloads/Raw_data/MNGU0/ema/*.ema"

files = glob(path)
num_files = len(files)
ratio = 0.8
num_train_files = int(np.ceil(ratio * num_files))
np.random.shuffle(files)

train_files = files[:num_train_files]
test_files = files[num_train_files:]


train_dest_ema = "/home/boomkin/repos/articulatory_inversion/mngu0_ema/train/"
train_dest_wav = "/home/boomkin/repos/articulatory_inversion/mngu0_wav/train/"

for ema_src in train_files:
    filename = ema_src.split("/")[-1]

    ema_dest = train_dest_ema + filename
    print("EMA transfer src:", ema_src, " dest: ", ema_dest)
    copyfile(ema_src, ema_dest)

    filename_wo_extension = filename.split(".")[0]
    wav_src = split(split(ema_src)[0])[0] + "/wav/" + filename_wo_extension + ".wav"
    wav_dest = train_dest_wav + filename_wo_extension + ".wav"

    print("WAV transfer src:", wav_src, " dest: ", wav_dest)
    copyfile(wav_src, wav_dest)

test_dest_ema = "/home/boomkin/repos/articulatory_inversion/mngu0_ema/test/"
test_dest_wav = "/home/boomkin/repos/articulatory_inversion/mngu0_wav/test/"

for ema_src in test_files:
    filename = ema_src.split("/")[-1]

    ema_dest = test_dest_ema + filename
    print("EMA transfer src:", ema_src, " dest: ", ema_dest)
    copyfile(ema_src, ema_dest)

    filename_wo_extension = filename.split(".")[0]
    wav_src = split(split(ema_src)[0])[0] + "/wav/" + filename_wo_extension + ".wav"
    wav_dest = test_dest_wav + filename_wo_extension + ".wav"

    print("WAV transfer src:", wav_src, " dest: ", wav_dest)
    copyfile(wav_src, wav_dest)



#copyfile(src, dst)
