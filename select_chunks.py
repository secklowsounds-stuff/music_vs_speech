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
import plac

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import data


def read_csv_files(path):
    # for each chunk name, for each class count the votes
    counts = defaultdict(lambda: defaultdict(lambda: 0))
    # just select taggers with name, there is an anonymous tagged.csv
    csv_files = path.glob('tagged_*.csv')
    n_voters = 0
    for file_path in csv_files:
        n_voters += 1
        with file_path.open('r') as f:
            reader = csv.reader(f, delimiter=',')
            rows = [(l[0], l[1]) for l in reader]
            # remove duplicate for same tagger (does not remove same voter voting different classes)
            rows = set(rows)
            for row in rows:
                chunk_name = row[0]
                class_name = row[1]
                counts[chunk_name][class_name] += 1

    return counts, n_voters


def filter_chunks(counts, n_voters, min_duration, required_count):
    print('initial #chunks:',len(counts.keys()))
    counts = count_based_filter(counts, n_voters, required_count)
    counts = duration_based_filter(counts, min_duration)

    return counts


def count_based_filter(counts, n_voters, required_count=None):
    # criterion 1: no 'unknown' votes
    counts = {chunk_name: chunk_counts for (chunk_name, chunk_counts) in counts.items()
              if not chunk_counts['unknown']}
    print('#chunks after count-based criterion 1 (no unknown votes):',len(counts.keys()))

    # criterion 2: at least 2/3 of people agree
    if not required_count:
        required_count = (n_voters * 2) // 3
    print('required_count:', required_count)
    counts = {chunk_name: chunk_counts for chunk_name, chunk_counts in counts.items()
              if any(class_count >= required_count for class_name, class_count in chunk_counts.items())}
    print('#chunks after count-based criterion 2 (at least 2/3 agree):',len(counts.keys()))

    return counts


def duration_based_filter(counts, min_duration):
    results = {}
    print('required minimum duration (seconds):', min_duration)
    # a list of all the .mp3 files
    chunk_files_locations = data.CHUNKS_LOCATION.glob('*.mp3')
    # select only the ones that match the previous selection
    chunk_files_locations = [chunk_location for chunk_location in chunk_files_locations
                             if any(chunk_location.name == chunk_name for chunk_name in counts.keys())]
    for chunk_location in tqdm(chunk_files_locations):
        audio, sample_rate = librosa.load(str(chunk_location))
        duration = librosa.get_duration(audio, sample_rate)
        chunk_name = chunk_location.name
        if duration > min_duration:
            results[chunk_name] = counts[chunk_name]
    print('#chunks after duration-based filter:', len(results.keys()))
    
    return results


def select_winners(counts):
    winners = [(chunk_name, max(chunk_counts.items(), key=operator.itemgetter(1))[
                0]) for chunk_name, chunk_counts in counts.items()]

    return winners


def write_winners(winners, location):
    with location.open('w') as f:
        writer = csv.writer(f)
        writer.writerows(winners)


@plac.annotations(
    min_duration=plac.Annotation("The minimum duration in seconds", 'option', 'd', metavar='SEC', type=int),
    required_count=plac.Annotation("How many voters have to agree, if None (default) is provided the value will be 2 of 3 of the number of voters", 'option', 'c', metavar='COUNT', type=int))
def main(min_duration=3, required_count=None):
    counts, n_voters = read_csv_files(data.DOCS_LOCATION)
    counts = filter_chunks(counts, n_voters, min_duration, required_count)
    winners = select_winners(counts)
    counts = defaultdict(lambda: 0)
    for _, label in winners:
        counts[label] += 1
    print(*counts.items())
    write_winners(winners, data.GOLD_LOCATION)

if __name__ == '__main__':
    plac.call(main)
