"""
Rules: chunks are selected for the gold standard if:

- count based:
    - no one tagged them as unknown
    - at least floor(2/3) of people tagged them in the same group (music/speech)
- duration based:
    - chunk duration at least 3 seconds (TODO agree on some value)
"""

import csv
import librosa
import operator

from collections import defaultdict
from pathlib import Path

import data

MIN_DURATION = 3


def read_csv_files(path):
    # for each chunk name, for each class count the votes
    counts = defaultdict(lambda: defaultdict(lambda: 0))
    csv_files = path.glob('*.csv')
    n_voters = 0
    for file_path in csv_files:
        n_voters += 1
        with file_path.open('r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                chunk_name = row[0]
                class_name = row[1]
                counts[chunk_name][class_name] += 1

    return counts, n_voters


def filter_chunks(counts, n_voters):
    print('initial #chunks:',len(counts.keys()))
    counts = count_based_filter(counts, n_voters)
    counts = duration_based_filter(counts)

    return counts


def count_based_filter(counts, n_voters):
    # criterion 1: no 'unknown' votes
    counts = {chunk_name: chunk_counts for (chunk_name, chunk_counts) in counts.items()
              if not chunk_counts['unknown']}
    print('#chunks after count-based criterion 1:',len(counts.keys()))

    # criterion 2: at least 2/3 of people agree
    required_count = (n_voters * 2) // 3
    print('required_count:', required_count)
    counts = {chunk_name: chunk_counts for chunk_name, chunk_counts in counts.items()
              if any(class_count >= required_count for class_name, class_count in chunk_counts.items())}
    print('#chunks after count-based criterion 2:',len(counts.keys()))

    return counts


def duration_based_filter(counts):
    results = {}
    # a list of all the .mp3 files
    chunk_files_locations = data.CHUNKS_LOCATION.glob('*.mp3')
    # select only the ones that match the previous selection
    chunk_files_locations = [chunk_location for chunk_location in chunk_files_locations
                             if any(chunk_location.name == chunk_name for chunk_name in counts.keys())]
    for chunk_location in chunk_files_locations:
        # if does not work, do str(chunk)
        audio, sample_rate = librosa.load(str(chunk_location))
        duration = librosa.get_duration(audio, sample_rate)
        chunk_name = chunk_location.name
        if duration > MIN_DURATION:
            results[chunk_name] = counts[chunk_name]
    print('#chunks after duration-based filter:', len(counts.keys()))
    
    return counts


def select_winners(counts):
    winners = [(chunk_name, max(chunk_counts.items(), key=operator.itemgetter(1))[
                0]) for chunk_name, chunk_counts in counts.items()]

    return winners


def write_winners(winners, location):
    with location.open('w') as f:
        writer = csv.writer(f)
        writer.writerows(winners)


def main():
    counts, n_voters = read_csv_files(data.DOCS_LOCATION)
    counts = filter_chunks(counts, n_voters)
    winners = select_winners(counts)
    write_winners(winners, data.GOLD_LOCATION)


if __name__ == '__main__':
    main()
