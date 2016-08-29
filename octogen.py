import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora
from string import punctuation
from collections import defaultdict, Counter


class Data:

    def __init__(self):
        self._punkt = nltk.data.load('tokenizers/punkt/english.pickle')
        self._stopwords = set(stopwords.words('english')) | set(punctuation)
        self._dict = corpora.Dictionary()
        self._load()

    def get_clean(self):
        return [self.clean(t) for t in self._trd]

    def tokenize(self, text):
        return [w for s in self._punkt.tokenize(text) for w in word_tokenize(s)]

    def clean(self, tokens):
        return [t for t in map(str.lower, tokens) if t not in self._stopwords]

    def _load(self):
        print('Loading texts')
        self._trd, self._trd_map = self._load_document('data/train_dialogs.txt')
        self._trm, self._trm_map = self._load_document('data/train_missing.txt')
#        self._tsd, self._tsd_map = self._load_document('data/test_dialogs.txt')
        self._tsm = []
        with open('data/test_missing.txt') as f:
            for l in f.readlines():
                k, p = l.strip().split(' +++$+++ ')
                self._tsm.append(self.tokenize(p))
        self._dict.filter_extremes(no_below=3)

    def _load_document(self, path):
        documents = []
        mapping = {}
        i = 0
        last_id = None
        buf = []
        with open(path) as f:
            for l in f.readlines():
                id, p = l.strip().split(' +++$+++ ')
                if last_id != id:
                    if last_id is not None:
                        documents.append(buf)
                        buf = []
                        mapping[i] = last_id
                        i += 1
                    last_id = id
                tokens = self.tokenize(p)
                buf += tokens
                self._dict.add_documents([self.clean(tokens)])
        return documents, mapping


def main():
    nltk.download(['punkt', 'stopwords'])
    data = Data()


if __name__ == '__main__':
    main()
