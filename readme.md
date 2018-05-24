# SecklowSounds music/speech classifier

## Get the data

run `./download.sh`

## Get the chunks that satisfy selection

run `python select_chunks.py`

## Compute melgrams

run `python compute_melgrams.py`

## Train model

set image_dim_ordering == 'th'.You can set it at ~/.keras/keras.json
run `python train.py`

## TODO