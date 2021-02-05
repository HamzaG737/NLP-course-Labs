from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
from random import choices
from tqdm.notebook import tqdm
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize
from collections import defaultdict
import random
import re
import pickle


__authors__ = ["author1", "author2", "author3"]
__emails__ = [
    "fatherchristmoas@northpole.dk",
    "toothfairy@blackforest.no",
    "easterbunny@greenfield.de",
]


def preprocess(sentence):
    sentence = re.sub(
        "[^a-zA-Z0-9]", " ", sentence
    )  ## remove non alpha numerical  caracters
    sentence = re.sub(
        "[0-9]+", "NUMTOKEN", sentence
    )  ## create NUMTOKEN word for numbers
    return sentence


def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append(word_tokenize(preprocess(l.lower())))
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter="\t")
    pairs = zip(data["word1"], data["word2"], data["similarity"])
    return pairs


def check_if_number(word):
    return re.sub("[0-9]", "NUMTOKEN", word)


class SkipGram:
    def __init__(
        self, sentences, lr=1e-2, nEmbed=100, negativeRate=5, winSize=5, minCount=2
    ):
        self.w2id = {}  # word to ID mapping

        self.vocab = []  # list of valid words
        self.occurences = defaultdict(int)
        self.minCount = minCount
        self.get_vocab(sentences)
        self.trainset = self.subsample(sentences)
        ## Initialize weights
        self.W_ = np.random.normal(0, 1, (nEmbed, len(self.vocab)))
        self.C_ = np.random.normal(0, 1, (nEmbed, len(self.vocab)))
        ## learning rate
        self.lr_ = lr
        self.neg_ = negativeRate
        self.winSize = winSize
        self.loss = []
        ## random vector for unknown words in test time.
        self.random_vector = np.random.normal(0, 1, (nEmbed, 1))

    def get_vocab(self, sentences):
        """
        Create vocab , word to id dict and a dict of occurences
        """
        for sentence in sentences:
            for word in sentence:
                self.occurences[word] += 1

        self.occurences = dict(self.occurences)
        ## filter by mincount
        self.occurences = {
            k: v for k, v in self.occurences.items() if v >= self.minCount
        }
        self.vocab = list(self.occurences.keys())
        self.w2id = dict(zip(self.vocab, range(len(self.vocab))))

        total_occurences = sum(list(self.occurences.values()))  ## normalize
        self.occurences = {
            k: v / total_occurences for k, v in self.occurences.items()
        }  ## normalize

    def subsample(self, sentences):
        """
        This function subsamples words that are frequent in the dataset.
        """
        trainset_ = []
        for sentence in sentences:
            current = []
            for word in sentence:
                ## if not in vocab (filtered by min count)
                if word not in self.vocab:
                    continue
                ## This is the probability to keep the word. The formulation is taken from word2vec paper.
                prob = (np.sqrt(self.occurences[word] / 1e-3) + 1) * (
                    1e-3 / self.occurences[word]
                )
                k = random.random()
                if k < prob:
                    current.append(word)
            trainset_.append(current)
        return trainset_

    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""
        l = list(self.w2id.values())
        for x in omit:
            l.remove(x)  ## faster than list comprehension
        return choices(l, k=self.neg_)

    def onehot(self, Id_):
        """
        One hot code labeling of the word indexes.
        """
        ## in case we want to encode one index
        if isinstance(Id_, int):
            m = np.zeros((len(self.vocab), 1))
            m[Id_ - 1] = 1
            return m
        else:  ## multiple indexes
            m = np.zeros((len(self.vocab), len(Id_)))
            for k, index in enumerate(Id_):
                m[index - 1, k] = 1
            return m

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self):
        self.trainWords, self.acc = 0, 0
        for counter, sentence in tqdm(
            enumerate(self.trainset), total=len(self.trainset)
        ):
            sentence = list(filter(lambda word: word in self.vocab, sentence))

            for wpos, word in enumerate(sentence):
                wIdx = self.w2id[word]
                winsize = np.random.randint(self.winSize) + 1
                start = max(0, wpos - winsize)
                end = min(wpos + winsize + 1, len(sentence))

                for context_word in sentence[start:end]:
                    ctxtId = self.w2id[context_word]
                    if ctxtId == wIdx:
                        continue
                    negativeIds = self.sample({wIdx, ctxtId})
                    self.trainWord(wIdx, ctxtId, negativeIds)
                    self.trainWords += 1

            if counter % 100 == 0:
                print(" > training %d of %d" % (counter, len(self.trainset)))
                acc_norm = self.acc / ((1 + self.neg_) * self.trainWords)
                print("loss : ", acc_norm)
                self.loss.append(acc_norm)
                self.trainWords = 0
                self.acc = 0.0

    def trainWord(self, wordId, contextId, negativeIds):

        y_context = self.onehot(contextId)
        z_negatives = self.onehot(negativeIds)

        ## faster than self.W_ @ onehotencoding
        x_w = self.W_[:, wordId - 1].reshape(-1, 1)
        y_c = self.C_[:, contextId - 1].reshape(-1, 1)
        Z_c = self.C_[:, [neg_id - 1 for neg_id in negativeIds]]
        ## compute gradients
        negative_grad_W = np.zeros((self.W_.shape[0], 1))
        negative_grad_C = np.zeros((1, self.C_.shape[1]))

        sig_neg = self.sigmoid(x_w.T @ Z_c)
        negative_grad_W = -np.sum(sig_neg * Z_c, axis=1).reshape(-1, 1)
        negative_grad_C = -np.sum(sig_neg * z_negatives, axis=1).reshape(1, -1)
        self.acc -= np.sum(np.log(1 - sig_neg + 1e-6), axis=1)[0]

        sig_pos = self.sigmoid(-x_w.T @ y_c)[0][0]
        self.acc -= np.log(1 - sig_pos + 1e-6)
        grad_w = np.zeros_like(self.W_)
        ## update only gradient for wordId token
        grad_w[:, wordId - 1] = (sig_pos * y_c + negative_grad_W).squeeze()
        grad_c = x_w @ (sig_pos * y_context.T + negative_grad_C)
        ## update weights
        self.W_ += self.lr_ * grad_w  ## gradient ascent since we want the argmax
        self.C_ += self.lr_ * grad_c

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def similarity(self, word1, word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        if word1 not in self.vocab:
            x_w = self.random_vector
        else:
            id1 = self.w2id[word1]
            x_w = self.W_ @ self.onehot(id1)

        if word2 not in self.vocab:
            x_c = self.random_vector
        else:
            id2 = self.w2id[word2]
            x_c = self.C_ @ self.onehot(id2)

        return self.sigmoid(x_w.T @ x_c)[0][0]

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="path containing training data", required=True)
    parser.add_argument(
        "--model",
        help="path to store/read model (when training/testing)",
        required=True,
    )
    parser.add_argument("--test", help="enters test mode", action="store_true")

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences, lr=1e-2, nEmbed=100, negativeRate=5)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a, b, sim_true in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print("%.5f" % (sg.similarity(a, b)))
