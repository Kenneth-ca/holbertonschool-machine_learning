#!/usr/bin/env python3
"""
Calculates the unigram BLEU score for a sentence
"""
import numpy as np


def n_grams(sentence, n):
    """
    Creates the n-grams from sentence
    :param sentence: a list containing the model proposed sentence
    :param n: the size of the n-gram to use for evaluation
    :return: the n-gram
    """
    list_grams_cand = []
    for i in range(len(sentence)):
        last = i + n
        begin = i
        if last >= len(sentence) + 1:
            break
        aux = sentence[begin: last]
        result = ' '.join(aux)
        list_grams_cand.append(result)
    return list_grams_cand


def ngram_bleu(references, sentence, n):
    """
    Calculates the unigram BLEU score for a sentence
    :param references: a list of reference translations
    each reference translation is a list of the words in the translation
    :param sentence: a list containing the model proposed sentence
    :param n: the size of the n-gram to use for evaluation
    :return: unigram BLEU score
    """
    grams = list(set(n_grams(sentence, n)))
    len_g = len(grams)
    reference_grams = []
    words_dict = {}

    for reference in references:
        list_grams = n_grams(reference, n)
        reference_grams.append(list_grams)

    for ref in reference_grams:
        for word in ref:
            if word in grams:
                if word not in words_dict.keys():
                    words_dict[word] = ref.count(word)
                else:
                    actual = ref.count(word)
                    prev = words_dict[word]
                    words_dict[word] = max(actual, prev)

    candidate = len(sentence)
    prob = sum(words_dict.values()) / len_g

    best_match = []
    for reference in references:
        ref_len = len(reference)
        diff = abs(ref_len - candidate)
        best_match.append((diff, ref_len))

    sort_tuple = sorted(best_match, key=(lambda x: x[0]))
    best = sort_tuple[0][1]
    if candidate > best:
        bleu = 1
    else:
        bleu = np.exp(1 - (best / candidate))
    score = bleu * np.exp(np.log(prob))
    return score
