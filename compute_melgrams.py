import csv
import pathlib
import numpy as np

import data


def main():
    with data.GOLD_LOCATION.open('r') as gold_file:
        reader = csv.reader(gold_file, delimiter=',')
        for row in reader:
            chunk_name = row[0]
            class_name = row[1]
            chunk_location = data.CHUNKS_LOCATION / chunk_name

            melgram = data.compute_melgram(str(chunk_location))
            #print(melgram.shape)

            output_folder = data.MELGRAM_LOCATION / class_name
            output_folder.mkdir(parents=True, exist_ok=True)
            np.save(output_folder / chunk_name, melgram)

if __name__ == '__main__':
    main()