"""
Load an MFCCS and visualize it
"""

import librosa
import librosa.display
import plac

import numpy as np

import matplotlib.pyplot as plt

def main(path):
    mfccs = np.load(path)
    # consider only one channel, MFCC is mono
    mfccs = mfccs[0]
    print(mfccs.shape)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plac.call(main)