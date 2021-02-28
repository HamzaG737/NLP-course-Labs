import pandas as pd
import spacy
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


nlp = spacy.load('en_core_web_sm')

def read(filename='TP2/data/traindata.csv', nlp=nlp):
    dataset = pd.read_csv(filename, sep='\t', header=None)
    dataset = dataset.rename(index=str, columns={ 0: "sentiment",
                                                  1: "aspect_category",
                                                  2: "target_term",
                                                  3: "position",
                                                  4: "review"})
    dataset.review = dataset.review.str.lower()
    dataset.review = dataset.review.str.replace('-', ' ')
    dataset.target_term = dataset.target_term.str.lower()
    dataset.target_term = dataset.target_term.str.replace('-', ' ')
    sentiment_terms = []
    for review in nlp.pipe(dataset['review']):
            if review.has_annotation("DEP"):
                sentiment_terms.append([token.lemma_ for token in review if
                                        (not token.is_stop and not token.is_punct and
                                         (token.pos_ in ["ADJ", "VERB", "INTJ", "ADV"]))])
            else:
                sentiment_terms.append([])
    dataset['sentiment_terms'] = sentiment_terms

    return dataset

trainset = read(filename='TP2/data/traindata.csv')
devset = read(filename='TP2/data/devdata.csv')

le = LabelEncoder()
le.fit(trainset['sentiment'])
trainset['sentiment_label'] = le.transform(trainset['sentiment'])
devset['sentiment_label'] = le.transform(devset['sentiment'])


import gensim.downloader as api
twitter_model = api.load('glove-twitter-50')

import networkx as nx
from tqdm import tqdm
def get_features(dataset=trainset, model=twitter_model):
    features = np.zeros((len(dataset), 50))
    for i in tqdm(range(len(dataset))):
        sentiment_terms = dataset.iloc[i, :]['sentiment_terms']
        target_term = dataset.iloc[i, :]['target_term']
        review = dataset.iloc[i, :]['review'].lower()
        edges = []
        for token in nlp(review):
            for child in token.children:
                edges.append(('{0}'.format(token.lower_),
                              '{0}'.format(child.lower_)))

        graph = nx.Graph(edges)
        #print(review)
        feats = np.zeros(50)
        coefs = []
        for term in sentiment_terms:
            try:
                feat = model.wv[nlp(term)[0].lemma_]
                min_distance = 100
                for target_subterm in nlp(target_term):
                    try:
                        dist = nx.shortest_path_length(graph, source=target_subterm.text.lower(), target=term)
                    except:
                        dist = 100
                    min_distance = min(dist, min_distance)
                coef = 1 / (min_distance + 1)**(0.2)
                coefs.append(coef)
                feats += coef * feat
            except:
                pass
        if len(coefs)>0:
            sum_coefs = np.array(coefs).sum()
            feats /= sum_coefs
            features[i, :] = feats
        '''if len(sentiment_terms) == 0:
            features[i, :] = np.zeros(50)
        else:
            l = []
            for term in sentiment_terms:
                try:
                    l.append(model.wv[term])
                except:
                    pass
            if len(l)>0:
                l = np.array(l)
                features[i, :] = l.mean(axis=0)
            else:
                features[i, :] = np.zeros(50)'''
        #features.append(feats)
    return np.array(features)

train_features = get_features(trainset, twitter_model)
dev_features = get_features(devset, twitter_model)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
enc.fit(trainset['aspect_category'].values.reshape(-1, 1))
tr_as_categ = enc.transform(trainset['aspect_category'].values.reshape(-1, 1))
dev_as_categ = enc.transform(devset['aspect_category'].values.reshape(-1, 1))

train_features = np.concatenate([train_features, tr_as_categ], axis=1)
dev_features = np.concatenate([dev_features, dev_as_categ], axis=1)


clf = RandomForestClassifier(max_depth=15)
clf.fit(train_features, trainset['sentiment_label'])
y_train_pred = clf.predict(train_features)

y_dev_pred = clf.predict(dev_features)
(y_dev_pred == devset['sentiment_label']).mean()



from TP2.src.classifier import Classifier
clf = Classifier()
clf.train('TP2/data/traindata.csv')

y_pred = clf.predict('TP2/data/devdata.csv')