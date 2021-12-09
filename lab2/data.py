import os
import csv
import gzip
import zipfile
import requests
import pandas as pd

STS_URL = "https://sbert.net/datasets/stsbenchmark.tsv.gz"
SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
MULTINLI_URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
DATA_DIR = 'data'


class TextExample:
    def __init__(self, texts, label) -> None:
        self.texts = texts
        self.label = label
        
    def __repr__(self) -> str:
        return f"TextExample({self.texts}, label={self.label})"


def load_sts():
    path = download(STS_URL)
    data = dict(train=list(), dev=list(), test=list())
    with gzip.open(path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            sample = TextExample([row['sentence1'], row['sentence2']], score)
            data[row['split']].append(sample)
    return data


def load_snli(include_mismatched=False):
    data = dict(train=list(), dev=list(), test=list())
    for url in [SNLI_URL, MULTINLI_URL]:
        path = download(url)
        with zipfile.ZipFile(path) as f:
            txt_files = [fname for fname in f.namelist() if fname.endswith('.txt')]
            for fname in txt_files:
                try:
                    type = next(t for t in data if t in fname)
                    assert include_mismatched or 'mismatched' not in fname
                except:
                    continue
                df = pd.read_csv(f.open(fname), sep='\t', quoting=csv.QUOTE_NONE)
                data[type].extend(TextExample([t.sentence1, t.sentence2], t.gold_label)
                                  for t in df.itertuples() if t.gold_label != '-')
    return data


def download(url, path=None):
    if path is None:
        path = os.path.join(DATA_DIR, os.path.basename(url))
    if not os.path.exists(path):
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)
    return path
