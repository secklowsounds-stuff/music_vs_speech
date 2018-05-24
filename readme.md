# SecklowSounds music/speech classifier

## Requirements

System:
- python2 / python3 (preferred)
- python2-pip / python3-pip (preferred)
- ffmpeg (as backend to librosa)

pip packages
- virtualenv (strongly advised)
- `pip install -r requirements.txt`

## Get the data

run `./download_all.sh` to download all, also the audio files, or `./download_csv` to download only the annotations.

## Get the chunks that satisfy selection

run `python select_chunks.py`

## Compute melgrams

run `python compute_melgrams.py`

## Train model

run `python train.py MODEL_NAME` with MODEL_NAME one of the following:
- `cnn`
- `small_cnn`
- `crnn`

## TODO