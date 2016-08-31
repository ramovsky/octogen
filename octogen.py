import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordNERTagger
from gensim import corpora, models, similarities
from string import punctuation
from collections import defaultdict, Counter


class Data:

    def __init__(self, prefix='train'):
        self._prefix = prefix
        self._punkt = nltk.data.load('tokenizers/punkt/english.pickle')
        self._stopwords = set(stopwords.words('english')) | set(punctuation)
        self.dictionary = corpora.Dictionary()
        self._load()

    def get_corpus(self):
        return [self.dictionary.doc2bow(self.clean(t)) for t in self._dialogs]

    def tokenize(self, text):
        return [w for s in self._punkt.tokenize(text) for w in word_tokenize(s)]

    def clean(self, tokens):
        return [t for t in map(str.lower, tokens) if t not in self._stopwords]

    def _load(self):
        print('Loading {} texts...'.format(self._prefix))
        self._dialogs, self.dmap = self._load_dialogs()
        self.missing = []
        self._mmap = {}
        with open('data/{}_missing.txt'.format(self._prefix)) as f:
            for i, l in enumerate(f.readlines()):
                k, p = l.strip().split(' +++$+++ ')
                tokens = self.tokenize(p)
                self.missing.append(tokens)
                self.dictionary.add_documents([self.clean(tokens)])
                self._mmap[i] = k
        self.dictionary.filter_extremes(no_below=2)

    def _load_dialogs(self):
        documents = []
        mapping = {}
        i = 0
        last_id = None
        buf = []
        with open('data/{}_dialogs.txt'.format(self._prefix)) as f:
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
                self.dictionary.add_documents([self.clean(tokens)])
        return documents, mapping


def main():
    nltk.download(['punkt', 'stopwords'])
    data = Data('test')
    corpus = data.get_corpus()
    tfidf = models.TfidfModel(corpus)
    print('Building TF-IDF index...')
    t_index = similarities.MatrixSimilarity(tfidf[corpus], num_best=10)
    print('Builing LDA index...')
    lda = models.LdaMulticore(corpus, id2word=data.dictionary, num_topics=40)
    l_index = similarities.MatrixSimilarity(lda[corpus], num_best=10)
    print('Idexies built')

    out = 'test_missing_with_predictions.txt'
    print('Saving output to {!r}'.format(out))
    with open(out, 'w') as f:
        for miss in data.missing:
            res = defaultdict(float)
            vector = data.dictionary.doc2bow(data.clean(miss))
            q = tfidf[vector]
            ql = lda[vector]
            for i, p in t_index[q]:
                res[i] += p
            for i, p in l_index[ql]:
                res[i] += p
            rating = sorted(res, key=res.get, reverse=True)
            id = data.dmap[rating[0]]
            line = '{} +++$+++ {}\n'.format(id, ' '.join(miss))
            f.write(line)


if __name__ == '__main__':
    main()
