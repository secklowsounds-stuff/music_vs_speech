import librosa
import numpy as np
from pathlib import Path

# common parameters
DATA_LOCATION = Path('datasets/SecklowSounds')
DOCS_LOCATION = DATA_LOCATION / 'docs'
CHUNKS_LOCATION = DOCS_LOCATION / 'chunks'
GOLD_LOCATION = DATA_LOCATION / 'gold.csv'
MELGRAM_LOCATION = DATA_LOCATION / 'melgrams'

# mel-spectrogram parameters
SR = 12000
N_FFT = 512
N_MELS = 96
HOP_LEN = 256
DURA = 29.12  # to make it 1366 frame..
N_FRAMES = 1366 # TODO discover where those numbers come from #int(DURA * SR)
# TODO see https://github.com/keras-team/keras/blame/5a48df22f0d9dd5b365ed5afc9923424d2e8c2c9/keras/layers/core.py#L76
# the shapes are ok float until the Dense layer

def compute_melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,96,1366), where
    96 == #mel-bins and 1366 == #time frame
    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load
    '''

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    # TODO remove padding / truncating, real variable length
    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)//2:(n_sample+n_sample_fit)//2]
    logam = librosa.power_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2,
                ref=1.0)
    ret = ret[np.newaxis, :]
    return ret