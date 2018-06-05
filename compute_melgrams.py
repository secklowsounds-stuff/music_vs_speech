import csv
import pathlib
import numpy as np

from tqdm import tqdm

import data


def main():
    with data.GOLD_LOCATION.open('r') as gold_file:
        reader = csv.reader(gold_file, delimiter=',')
        row_count = sum(1 for row in open(data.GOLD_LOCATION))
        for row in tqdm(reader, total=row_count):
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