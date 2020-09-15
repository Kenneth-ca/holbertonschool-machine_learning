#!/usr/bin/env python3

from gensim.test.utils import common_texts
word2vec_model = __import__('2-word2vec').word2vec_model
gensim_to_keras = __import__('3-gensim_to_keras').gensim_to_keras

print(common_texts[:2])
w2v = word2vec_model(common_texts, min_count=1)
print(gensim_to_keras(w2v))
