from dataclasses import dataclass, field
from collections import Counter
import os
import pickle

import h5py
import numpy as np
from tqdm import tqdm

from vectorizer import Vectorizer


@dataclass
class CooccurrenceEntries:
    vectorized_corpus: list
    vectorizer: Vectorizer

    @classmethod
    def setup(cls, corpus, vectorizer):
        return cls(
            vectorized_corpus=vectorizer.vectorize(corpus),
            vectorizer=vectorizer
        )

    def validate_index(self, index, lower, upper):
        is_unk = index == self.vectorizer.vocab.unk_token
        if lower < 0:
            return not is_unk
        return not is_unk and index >= lower and index <= upper

    def build(
        self,
        window_size,
        num_partitions,
        chunk_size,
        output_directory="."
    ):
        vocab_len = len(self.vectorizer.vocab)
        partition_step = vocab_len // num_partitions
        split_points = [0]
        while split_points[-1] + partition_step <= vocab_len:
            split_points.append(split_points[-1] + partition_step)
        split_points[-1] = vocab_len

        for partition_id in tqdm(range(len(split_points) - 1)):
            index_lower = split_points[partition_id]
            index_upper = split_points[partition_id + 1] - 1
            cooccurr_counts = Counter()
            for i in tqdm(range(len(self.vectorized_corpus)), leave=False):
                if self.vectorized_corpus[i] == vocab_len: continue     # [END] token
                if not self.validate_index(
                    self.vectorized_corpus[i],
                    index_lower,
                    index_upper
                ):
                    continue

                context_lower = max(i - (window_size // 2), 0)
                context_upper = min(i + (window_size // 2) + 1, len(self.vectorized_corpus))

                if vocab_len in self.vectorized_corpus[context_lower: i]:
                    context_lower = self.vectorized_corpus.index(vocab_len, context_lower, i) + 1
                if vocab_len in self.vectorized_corpus[i+1: context_upper]:
                    context_upper = self.vectorized_corpus.index(vocab_len, i+1, context_upper)

                for j in range(context_lower, context_upper):
                    if i == j or not self.validate_index(
                        self.vectorized_corpus[j],
                        -1,
                        -1
                    ):
                        continue
                    cooccurr_counts[(self.vectorized_corpus[i], self.vectorized_corpus[j])] += 1 / abs(i - j)

            cooccurr_dataset = np.zeros((len(cooccurr_counts), 3))
            for index, ((i, j), cooccurr_count) in enumerate(cooccurr_counts.items()):
                cooccurr_dataset[index] = (i, j, cooccurr_count)
            if partition_id == 0:
                file = h5py.File(
                    os.path.join(
                        output_directory,
                        "cooccurrence.hdf5"
                    ),
                    "w"
                )
                dataset = file.create_dataset(
                    "cooccurrence",
                    (len(cooccurr_counts), 3),
                    maxshape=(None, 3),
                    chunks=(chunk_size, 3)
                )
                prev_len = 0
            else:
                prev_len = dataset.len()
                dataset.resize(dataset.len() + len(cooccurr_counts), axis=0)
            dataset[prev_len: dataset.len()] = cooccurr_dataset

        file.close()
        with open(os.path.join(output_directory, "vocab.pkl"), "wb") as file:
            pickle.dump(self.vectorizer.vocab, file)
