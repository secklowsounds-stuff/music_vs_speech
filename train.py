import plac

import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from music_tagger_cnn import MusicTaggerCNN, SmallCNN, SmallestCNN
from music_tagger_crnn import MusicTaggerCRNN

import data

K.set_image_dim_ordering("th")

def load_data(path):
    x = []
    y = []

    class_names = []

    for class_id, subfolder in enumerate([el for el in path.iterdir() if el.is_dir()]):
        class_names.append(subfolder.name)
        for melgram_path in subfolder.glob('*.npy'):
            melgram = np.load(melgram_path)
            x.append(melgram)
            y.append(class_id)

    y = to_categorical(y, len(class_names))

    return np.array(x), np.array(y), class_names


def main(net_type, epochs=10):
    x, y, class_names = load_data(data.MELGRAM_LOCATION)
    print(class_names)

    n_classes = len(class_names)

    if net_type == 'cnn':
        model = MusicTaggerCNN(data.N_FRAMES, data.N_MELS, n_classes)
    elif net_type == 'small_cnn':
        model = SmallCNN(data.N_FRAMES, data.N_MELS, n_classes)
    elif net_type == 'smallest_cnn':
        model = SmallestCNN(data.N_FRAMES, data.N_MELS, n_classes)
    elif net_type == 'crnn':
        model = MusicTaggerCRNN(data.N_FRAMES, data.N_MELS, n_classes)

    else:
        raise ValueError(net_type)

    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # TODO change batch size
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    model.save('music_{}_epochs:{}.h5'.format(net_type, epochs))

    return

if __name__ == '__main__':
    plac.call(main)
