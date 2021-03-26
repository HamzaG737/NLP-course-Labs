import pandas as pd
import spacy
from sklearn.preprocessing import LabelEncoder
import gensim.downloader as api
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


class Classifier:
    """The Classifier"""
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.label_encoder = LabelEncoder()
        self.fasttext_model = api.load('glove-twitter-50')
        self.enc = OneHotEncoder(sparse=False)
        self.clf = RandomForestClassifier()

    def read(self, filename):
        dataset = pd.read_csv(filename, sep='\t', header=None)
        dataset = dataset.rename(index=str, columns={0: "sentiment",
                                                     1: "aspect_category",
                                                     2: "target_term",
                                                     3: "position",
                                                     4: "review"})
        dataset.review = dataset.review.str.lower()
        dataset.review = dataset.review.str.replace('-', ' ')
        dataset.target_term = dataset.target_term.str.lower()
        dataset.target_term = dataset.target_term.str.replace('-', ' ')
        sentiment_terms = []
        for review in self.nlp.pipe(dataset['review']):
            if review.has_annotation("DEP"):
                sentiment_terms.append([token.lemma_ for token in review if
                                        (not token.is_stop and not token.is_punct and
                                         (token.pos_ in ["ADJ", "VERB", "INTJ", "ADV"]))])
            else:
                sentiment_terms.append([])
        dataset['sentiment_terms'] = sentiment_terms

        return dataset

    def get_features(self, dataset, model):
        features = np.zeros((len(dataset), 50))
        for i in tqdm(range(len(dataset))):
            sentiment_terms = dataset.iloc[i, :]['sentiment_terms']
            target_term = dataset.iloc[i, :]['target_term']
            review = dataset.iloc[i, :]['review'].lower()
            edges = []
            for token in self.nlp(review):
                for child in token.children:
                    edges.append(('{0}'.format(token.lower_),
                                  '{0}'.format(child.lower_)))

            graph = nx.Graph(edges)
            feats = np.zeros(50)
            coefs = []
            for term in sentiment_terms:
                try:
                    feat = model.wv[self.nlp(term)[0].lemma_]
                    min_distance = 100
                    for target_subterm in self.nlp(target_term):
                        try:
                            dist = nx.shortest_path_length(graph, source=target_subterm.text.lower(), target=term)
                        except:
                            dist = 100
                        min_distance = min(dist, min_distance)
                    coef = 1 / (min_distance + 1) ** (0.2)
                    coefs.append(coef)
                    feats += coef * feat
                except:
                    pass
            if len(coefs) > 0:
                sum_coefs = np.array(coefs).sum()
                feats /= sum_coefs
                features[i, :] = feats
        return features

    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        self.trainset = self.read(filename=trainfile)
        self.label_encoder.fit(self.trainset['sentiment'])
        self.trainset['sentiment_label'] = self.label_encoder.transform(self.trainset['sentiment'])
        self.train_features = self.get_features(self.trainset, self.fasttext_model)
        self.enc.fit(self.trainset['aspect_category'].values.reshape(-1, 1))
        onehot = self.enc.transform(self.trainset['aspect_category'].values.reshape(-1, 1))
        self.train_features = np.concatenate([self.train_features, onehot], axis=1)
        self.clf.fit(self.train_features, self.trainset['sentiment_label'])

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        self.testset = self.read(filename=datafile)
        self.testset['sentiment_label'] = self.label_encoder.transform(self.testset['sentiment'])
        self.test_features = self.get_features(self.testset, self.fasttext_model)
        onehot = self.enc.transform(self.testset['aspect_category'].values.reshape(-1, 1))
        self.test_features = np.concatenate([self.test_features, onehot], axis=1)
        pred_labels = self.clf.predict(self.test_features)
        pred_labels = list(self.label_encoder.inverse_transform(list(pred_labels)))
        return pred_labels




