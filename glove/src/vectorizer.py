from dataclasses import dataclass

from vocabulary import Vocabulary


@dataclass
class Vectorizer:
    vocab: Vocabulary

    @classmethod
    def from_corpus(cls, corpus, vocab_size=None, min_freq=None):
        vocab = Vocabulary()
        for token in corpus:
            if token != '[END]': vocab.add(token)
        if vocab_size != None and min_freq == None:
            vocab_subset = vocab.get_topk_subset(vocab_size)
        elif min_freq != None and vocab_size == None:
            vocab_subset = vocab.get_subset_by_freq(min_freq)
        else:
            raise ValueError(f'Either vocab_size or min_freq should be set in config | provided: vocab_size={vocab_size}, min_freq:{min_freq}')
        vocab_subset.shuffle()
        return cls(vocab_subset)

    def vectorize(self, corpus):
        # return [self.vocab[token] for token in corpus]
        vectorized_corpus = []
        for token in corpus:
            if token != '[END]': vectorized_corpus.append(self.vocab[token])
            else: vectorized_corpus.append(len(self.vocab))
        return vectorized_corpus
