import pandas as pd
import re
import time
import xml.etree.ElementTree

from collections import OrderedDict
from matplotlib import pyplot
from nltk.tokenize.casual import TweetTokenizer
from numpy import array as np_array
from sklearn import metrics

TWEET_TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
EMB_SEP_CHAR = " "
RE_TOKEN_USER = re.compile(
    r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)")


def tokenize(text):
    """Tokenize an input text
    
    Args:
        text: A String with the text to tokenize
    
    Returns:
        A list of Strings (tokens)
    """
    text_tokenized = TWEET_TOKENIZER.tokenize(text)
    return text_tokenized


def fit_transform_vocabulary(corpus):
    """Creates the vocabulary of the corpus
    
    Args:
        corpus: A list of str (documents)
        
    Returns:
        A tuple whose first element is a dictionary word-index and the second
        element is a list of list in which each position is the index of the 
        token in the vocabulary
    """

    vocabulary = {}
    corpus_indexes = []
    corpus_indexes_append = corpus_indexes.append
    index = 2
    for doc in corpus:
        doc_indexes = []
        tokens = tokenize(doc)
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = index
                doc_indexes.append(index)
                index += 1
            else:
                doc_indexes.append(vocabulary[token])

        corpus_indexes_append(doc_indexes)

    return (vocabulary, corpus_indexes)


def fit_transform_vocabulary_pretrain_embeddings(corpus, pre_embeddings_index):
    """Creates the vocabulary of the corpus.
        Index 0: padding
        Index 1: OOV.
    
    Args:
        corpus: A list os str (documents)
        
    Returns:
        A tuple whose first element is a dictionary word-index and the second
        element is a list of list in which each position is the index of the 
        token in the vocabulary
    """

    vocabulary = {}
    corpus_indexes = []
    corpus_indexes_append = corpus_indexes.append
    index = 0
    own_lowercase = str.lower
    for doc in corpus:
        doc_indexes = []
        tokens = tokenize(own_lowercase(doc))
        for token in tokens:
            if RE_TOKEN_USER.fullmatch(token):
                token = "@user"
            if token in pre_embeddings_index:
                index = pre_embeddings_index[token]
                doc_indexes.append(index)
                if token not in vocabulary:
                    vocabulary[token] = index
            else:
                index = 1
                doc_indexes.append(index)
                if token not in vocabulary:
                    vocabulary[token] = index
        corpus_indexes_append(doc_indexes)

    return (vocabulary, corpus_indexes)


def read_embeddings(path, offset):
    """Load embeddings file.
    """
    word_embeddings = [[] for i in range(offset)]
    word_indexes = {}
    with open(path, "r", encoding="utf-8") as emb_file:
        emb_file.readline()
        for line in emb_file:
            fields = line.partition(EMB_SEP_CHAR)
            word = fields[0].strip()
            own_strip = str.strip
            emb_values = np_array([float(x) for x in own_strip(fields[-1]).split(EMB_SEP_CHAR)])
            word_indexes[word] = len(word_embeddings)
            word_embeddings.append(emb_values)

    return (word_embeddings, word_indexes)


def evaluate(real_ys, predicted_ys, model_name, classes_index):
    accuracy = metrics.accuracy_score(real_ys, predicted_ys)
    macro_precision = metrics.precision_score(real_ys, predicted_ys,
                                              labels=classes_index, average="macro")
    macro_recall = metrics.recall_score(real_ys, predicted_ys,
                                        labels=classes_index, average="macro")
    macro_f1 = metrics.f1_score(real_ys, predicted_ys,
                                labels=classes_index, average="macro")
    micro_f1 = metrics.f1_score(real_ys, predicted_ys,
                                labels=classes_index, average="micro")

    print("*** Results " + model_name + " ***")
    print("Accuracy: " + str(accuracy))
    print("Macro-Precision: " + str(macro_precision))
    print("Macro-Recall: " + str(macro_recall))
    print("Macro-F1: " + str(macro_f1))
    print("Micro-F1: " + str(micro_f1))

    df = pd.DataFrame(OrderedDict({'accuracy': accuracy,
                                   'macro_precision': macro_precision,
                                   'macro_recall': macro_recall,
                                   'macro_f1': macro_f1,
                                   'micro_f1': micro_f1}), index=[0])

    df.rename(index={0: model_name}, inplace=True)

    return df


def kaggle_file(ids, ys, model, NUM_TO_CLASSES_DIC):
    real_classes = [NUM_TO_CLASSES_DIC[y] for y in ys]
    df = pd.DataFrame(OrderedDict({'Id': ids, 'Expected': real_classes}))
    file_name = './data/results/' + model + '-' + str(int(time.time())) + '.csv'
    df.to_csv(file_name, index=False)


def get_raw_tweets(dataset):
    tree = xml.etree.ElementTree.parse(dataset)
    root = tree.getroot()
    return root.findall('tweet')


class Tweet(object):
    def __init__(self, id, x, y=None):
        self.id = id
        self.x = x
        self.y = y


def get_ids(tweets):
    return [t.id for t in tweets]


def get_xs(tweets):
    return [t.x for t in tweets]


def get_ys(tweets):
    return [t.y for t in tweets]


def plot_graphic(history, name):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title(name)
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Validation'], loc='upper right')
    pyplot.savefig(name + '-' + str(int(time.time())) + '.png')
