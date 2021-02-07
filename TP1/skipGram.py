from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
from random import choices
from tqdm.notebook import tqdm
import spacy

nlp = spacy.load("en")

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
    sentence = re.sub(" +", " ", sentence)  ## remove unnecessary spaces
    sentence = re.sub(
        "[0-9]+", "NUMTOKEN", sentence
    )  ## create NUMTOKEN word for numbers
    return sentence


def text2sentences(path):
    sentences = []
    with open(path) as f:
        for l in tqdm(f):
            sentence = []
            doc = nlp(preprocess(l.lower()))
            for token in doc:
                sentence.append(token.lemma_)
            sentences.append(sentence)
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter="\t")
    pairs = zip(data["word1"], data["word2"], data["similarity"])
    return pairs


def check_if_number(word):
    return re.sub("[0-9]", "NUMTOKEN", word)


class SkipGram:
    def __init__(
        self,
        sentences,
        path_save,
        lr=1e-2,
        nEmbed=100,
        negativeRate=5,
        winSize=5,
        minCount=2,
    ):
        self.path_save = path_save
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

        ## occurences for negative sampling
        sum_neg = np.sum(np.array(list(self.occurences.values())) ** (3 / 4))
        self.occurences_neg = {
            k: (v ** (3 / 4)) / sum_neg for k, v in self.occurences.items()
        }

    def subsample(self, sentences):
        """
        This function subsamples words that are frequent in the dataset.
        """
        trainset_ = []
        total_occurences = sum(list(self.occurences.values()))  ## normalize
        occurences_ratio = {
            k: v / total_occurences for k, v in self.occurences.items()
        }  ## normalize
        for sentence in sentences:
            current = []
            for word in sentence:
                ## if not in vocab (filtered by min count)
                if word not in self.vocab:
                    continue
                ## This is the probability to keep the word. The formulation is taken from word2vec paper.
                prob = (np.sqrt(occurences_ratio[word] / 1e-3) + 1) * (
                    1e-3 / occurences_ratio[word]
                )
                k = random.random()
                if k < prob:
                    current.append(word)
            trainset_.append(current)
        return trainset_

    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""

        ## we extract self.neg_ + 2 samples because in the worst case, two samples correspond to omit set.
        rand_ = np.random.multinomial(
            n=self.neg_ + 2, pvals=list(self.occurences_neg.values())
        )
        rand_ = list(np.where(rand_ >= 1)[0])
        rand_ = [id_ for id_ in rand_ if id_ not in omit]
        return rand_[: self.neg_]

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
                self.save(self.path_save)

    def trainWord(self, wordId, contextId, negativeIds):

        x_w = self.W_[:, wordId].reshape(-1, 1)
        y_c = self.C_[:, contextId].reshape(-1, 1)
        Z_c = self.C_[:, [neg_id for neg_id in negativeIds]]

        ## compute gradients
        sig_neg = self.sigmoid(x_w.T @ Z_c)
        negative_grad_W = np.zeros((self.W_.shape[0], 1))
        negative_grad_W = -np.sum(sig_neg * Z_c, axis=1).reshape(-1, 1)

        negative_grad_C = np.zeros((len(self.vocab), len(negativeIds)))
        for neg_id in negativeIds:
            negative_grad_C[neg_id, :] = sig_neg
        negative_grad_C = -np.sum(negative_grad_C, axis=1).reshape(1, -1)
        self.acc -= np.sum(np.log(1 - sig_neg + 1e-6), axis=1)[0]

        sig_pos = self.sigmoid(-x_w.T @ y_c)[0][0]
        self.acc -= np.log(1 - sig_pos + 1e-6)  ## add loss
        grad_w = np.zeros_like(self.W_)
        grad_w[:, wordId] = (sig_pos * y_c + negative_grad_W).squeeze()

        grad_c = np.zeros_like(self.C_)
        for neg_id in negativeIds:
            grad_c[:, neg_id] = x_w[:, 0] * negative_grad_C[:, neg_id]
        grad_c[:, contextId] = sig_pos * x_w[:, 0]

        ## update weights
        # only one column in W_ needs update, the rest are zeros
        self.W_[:, wordId] += (
            self.lr_ * grad_w[:, wordId]
        )  ## gradient ascent since we want the argmax
        # only modified context words will be updated
        for c_id in negativeIds + [contextId]:
            self.C_[:, c_id] += self.lr_ * grad_c[:, c_id]

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
            x_w = self.W_[id1]

        if word2 not in self.vocab:
            x_c = self.random_vector
        else:
            id2 = self.w2id[word2]
            x_c = self.W_[id2]

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
        sg = SkipGram(
            sentences, path_save=opts.model, lr=1e-2, nEmbed=100, negativeRate=5
        )
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print("%.5f" % (sg.similarity(a, b)))
