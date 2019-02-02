import joblib
from tabulate import tabulate
import numpy as np
import io
import collections

class Codebook(object):
    def __init__(self, tokens):
        self.tokens = tokens

    def token2idx(self, token):
        return self.tokens.index(token)

    def idx2token(self, idx):
        return self.tokens[idx]

    def encode(self, text):
        return [self.token2idx(token) for token in text]

    @property
    def size(self):
        return len(self.tokens)


mem = joblib.Memory('/tmp/dataset_util')


@mem.cache
def make_codebook(text):
    all_chars = list(sorted(set(text)))
    codebook = Codebook(all_chars)
    return codebook


@mem.cache
def get_zip_ratio(text):
    import zlib
    text = text.encode()
    smalltext = zlib.compress(text, level=-1)
    ratio = len(smalltext) / len(text)
    return ratio


def process_dataset(text_file, print_stats=True):
    with io.open(text_file, encoding='utf-8') as f:
        text = f.read().strip()
    codebook = make_codebook(text)
    if print_stats:
        token2count = collections.Counter(text)
        counts = np.array([token2count[c] for c in codebook.tokens])
        probs = counts / counts.sum()
        print(tabulate(zip(map(repr, codebook.tokens), probs, map(int, counts)),
                       headers=['tokens', 'probs', 'counts'], floatfmt='.3e'))
        zipratio = get_zip_ratio(text)
        print(tabulate([
            ('Marg ent', (probs * np.log(1 / probs)).sum()),
            ('Zip', zipratio * np.log(256))
        ]))
    return text, codebook

def iterbatches(*arrays, num_batches=None, batch_size=None, shuffle=True, include_final_partial_batch=True):
    assert (num_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(a.shape[0] == n for a in arrays[1:])
    inds = np.arange(n)
    if shuffle: np.random.shuffle(inds)
    sections = np.arange(0, n, batch_size)[1:] if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)
