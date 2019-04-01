import random
import re
import time
import xml.etree.ElementTree
from collections import OrderedDict

import keras
import matplotlib
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import tensorflow
from matplotlib import pyplot
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize.casual import TweetTokenizer
from num2words import num2words
from numpy import array as np_array
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from unidecode import unidecode

import src.util.global_vars

TWEET_TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
EMB_SEP_CHAR = " "
RE_TOKEN_USER = re.compile(
    r"(?<![A-Za-z0-9_!@#$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)")
FOLDS_CV = 5
BAD_WORDS = ["cabron", "cabrona", "mierda", "cojones", "joder", "tonto", "puto", "puta", "gilipollas", "hostia",
             "ostia", "follen", "follar", "coño", "cago", "cagar", "tonto", "tonta", "idiota", "estupido", "feo", "fea",
             "gordo", "gorda", "maldito", "maldita", "pudrete", "zorra", "imbecil", "baboso", "babosa", "besugo",
             "besufa", "brasas", "capullo", "capulla", "cenutrio", "cenutria", "ceporro", "ceporra", "cretino",
             "cretina", "gañan", "lameculos", "lerdo", "lerda", "palurdo", "palurda", "panoli", "pagafantas",
             "tocapelotas"]


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

    return vocabulary, corpus_indexes


def fit_transform_vocabulary_pretrain_embeddings(corpus, pre_embeddings_index):
    """
    Creates the vocabulary of the corpus.
        Index 0: padding
        Index 1: OOV.

    :param corpus: represents the documents, in our case the train dataset.
    :param pre_embeddings_index:

    Returns:
        A tuple whose first element is a dictionary word-index and the second
        element is a list of list in which each position is the index of the
        token in the vocabulary
    """

    vocabulary = {}  # vocabularies dict. associates foe each token in vocabulary, an index.
    corpus_indexes = []  # corpus indexes list
    corpus_indexes_append = corpus_indexes.append
    index = 0
    own_lowercase = str.lower  # lower case a string

    for doc in corpus:  # for each document in corpus. a document is a tweet in our case.
        doc_indexes = []  # the
        tokens = tokenize(own_lowercase(doc))  # tokens, is a list of tokenizing doc, also in lowercase.
        # so each word in doc will have a token corresponding to it.

        for token in tokens:  # for each token ..
            if RE_TOKEN_USER.fullmatch(token):  # if token fully match RE_TOKEN_USER, then replace token with " @user "
                token = "@user"

            if token in pre_embeddings_index:  # if token in exists in pre_embeddings_index.
                index = pre_embeddings_index[token]  # then save its index from pre_embeddings_index
            else:  # if token does not exist in pre_embeddings_index, the,
                index = 1  # assign 1 as index

            doc_indexes.append(
                index)  # add the corresponding index for token in doc_indexes. or better said the tweet indexes.

            if token not in vocabulary:  # if token, does not exist in vocabulary, then ..
                vocabulary[token] = index  # add the corresponding index for token into vocabulary.

        corpus_indexes_append(doc_indexes)  # add the doc_indexes into corpus_indexes

    return vocabulary, corpus_indexes


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

    return word_embeddings, word_indexes


def evaluate(real_ys, predicted_ys, model_name):
    """
    evaluate, is a function used to obtain the evaluation metrics for model_name with real_ys as the real labels and
    predicted_ys as the predicted labels.

    :param real_ys: the real labels
    :param predicted_ys: the predicted labels
    :param model_name: the model to evaluate
    :return:
    """
    macro_f1 = metrics.f1_score(real_ys, predicted_ys,
                                labels=list(src.util.global_vars.__CLASSES_TO_NUM_DIC__.values()), average="macro")
    micro_f1 = metrics.f1_score(real_ys, predicted_ys,
                                labels=list(src.util.global_vars.__CLASSES_TO_NUM_DIC__.values()), average="micro")

    df = pd.DataFrame(OrderedDict({'macro_f1': macro_f1, 'micro_f1': micro_f1}), index=[0])
    df.rename(index={0: model_name}, inplace=True)
    print(df)

    return df


def kaggle_file(ids, ys, model):
    """
    src.util.utilities.kaggle_file, a function used to prepare the submission file.

    :param ids: instances ID's
    :param ys: the predicted labels
    :param model: the model name
    :return:
    """
    real_classes = [src.util.global_vars.__NUM_TO_CLASSES_DIC__[y] for y in ys]
    df = pd.DataFrame(OrderedDict({'Id': ids, 'Expected': real_classes}))
    file_name = '../submissions/' + model + '-' + str(int(time.time())) + '.csv'
    df.to_csv(file_name, index=False)


def get_raw_tweets(dataset):
    """
    get_get_raw_tweets, a function used to read the tweets from the original source dataset. In this case it comes in
    XML format.

    :param dataset: the source dataset

    :return: a list of raw tweets
    """

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


def plot_graphic(histories, name):
    matplotlib.style.use('seaborn')
    colors = ['#00796b', '#43a047', '#1de9b6', '#00e676', '#76ff03']

    losses = []
    val_losses = []

    for i in range(5):
        losses.append(histories[i].history['loss'])
        val_losses.append(histories[i].history['val_loss'])

        pyplot.plot(histories[i].history['loss'], color=colors[i], linestyle='--')
        pyplot.plot(histories[i].history['val_loss'], color=colors[i], linestyle=':')

    pyplot.plot(np.mean(np.array(losses), axis=0), color='#d81b60', linestyle='--')
    pyplot.plot(np.mean(np.array(val_losses), axis=0), color='#d81b60', linestyle=':')

    train_loss = mlines.Line2D([], [], color='black', linestyle='--', marker='None', markersize=7, label='train loss')
    val_loss = mlines.Line2D([], [], color='black', linestyle=':', marker='None', markersize=7, label='val loss')

    train_loss_mean = mlines.Line2D([], [], color='#d81b60', linestyle='--', marker='None', markersize=7,
                                    label='train loss mean')
    val_loss_mean = mlines.Line2D([], [], color='#d81b60', linestyle=':', marker='None', markersize=7,
                                  label='val loss mean')

    green_1 = mlines.Line2D([], [], color='#00796b', marker='s', linestyle='None', markersize=7, label='partition 1')
    green_2 = mlines.Line2D([], [], color='#43a047', marker='s', linestyle='None', markersize=7, label='partition 2')
    green_3 = mlines.Line2D([], [], color='#1de9b6', marker='s', linestyle='None', markersize=7, label='partition 3')
    green_4 = mlines.Line2D([], [], color='#00e676', marker='s', linestyle='None', markersize=7, label='partition 4')
    green_5 = mlines.Line2D([], [], color='#76ff03', marker='s', linestyle='None', markersize=7, label='partition 5')

    pyplot.title(name)
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')

    pyplot.legend(
        handles=[train_loss, val_loss, train_loss_mean, val_loss_mean, green_1, green_2, green_3, green_4, green_5],
        loc='upper right')
    pyplot.savefig("../plots/" + name + '-' + str(int(time.time())) + '.eps', dpi=1000, format='eps')


def remove_emojis(tweet):
    # :), : ), :-), (:, ( :, (-:, :'), :D, : D, :-D, xD, x-D, XD, X-D, xd
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' POSITIVE POSITIVE POSITIVE POSITIVE ', tweet)
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D|xd)', ' POSITIVE POSITIVE POSITIVE POSITIVE ', tweet)
    # jajaja, lol
    tweet = re.sub(r'(a*ja+j[ja]*|o?l+o+l+[ol]*)', ' POSITIVE POSITIVE POSITIVE POSITIVE ', tweet)
    # hahaha
    tweet = re.sub(r'(a*ha+h[ha])', ' POSITIVE POSITIVE POSITIVE POSITIVE ', tweet)
    # <3, :*
    tweet = re.sub(r'(<3|:\*)', ' POSITIVE POSITIVE POSITIVE POSITIVE ', tweet)
    # ;-), ;), ;-D, ;D, (;,  (-;, ^^
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;|\^\^)', ' POSITIVE POSITIVE POSITIVE POSITIVE ', tweet)
    # :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' NEGATIVE NEGATIVE NEGATIVE NEGATIVE ', tweet)
    # :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' NEGATIVE NEGATIVE NEGATIVE NEGATIVE ', tweet)
    return tweet


def preprocess_tweets(tweets, stemming=False):
    # download('stopwords')
    # download('punkt')
    stop_words = stopwords.words('spanish')

    preprocessed_tweets = []
    # lower case and remove accent marks
    lowercase_tweets = [unidecode(tweet.lower()) for tweet in tweets]

    # stemming words
    stemmer = SnowballStemmer('spanish')

    for tweet in lowercase_tweets:
        # remove punctuation
        tweet = tweet.strip('\'"?!,.():;')
        # convert more than 2 letter repetitions to 2 letter
        tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
        # Remove - & '
        tweet = re.sub(r'(-|\')', '', tweet)

        # Replaces URLs with the word URL
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
        # Remove @handle
        tweet = re.sub(r'@[\S]+', '', tweet)
        # Replaces #hashtag with hashtag
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
        # Remove RT (retweet)
        tweet = re.sub(r'\brt\b', '', tweet)
        # Replace 2+ dots with space
        tweet = re.sub(r'\.{2,}', ' ', tweet)
        # Strip space, " and ' from tweet
        tweet = tweet.strip(' "\'')
        # Replace emojis with either POSTIIVE or NEGATIVE X4
        tweet = remove_emojis(tweet)
        # Replace multiple spaces with a single space
        tweet = re.sub(r'\s+', ' ', tweet)

        tweet_words = word_tokenize(tweet)

        if stemming:
            tweet_words = [stemmer.stem(word) for word in tweet_words]

        # Replace bad words by NEGATIVE x4
        tweet_words = [' NEGATIVE NEGATIVE NEGATIVE NEGATIVE ' if word in BAD_WORDS else word for word in tweet_words]

        # number to words
        tweet_words = [num2words(float(word), lang='es') if word.isnumeric() else word for word in
                       tweet_words]
        # remove stopwords
        tweet_words = ' '.join(word for word in tweet_words if word not in stop_words)

        preprocessed_tweets.append(tweet_words)

    return preprocessed_tweets


def micro_f1(y_true, y_pred):
    tp = keras.backend.sum(keras.backend.cast(y_true * y_pred, tensorflow.float32), axis=0)
    fp = keras.backend.sum(keras.backend.cast((1 - y_true) * y_pred, tensorflow.float32), axis=0)
    fn = keras.backend.sum(keras.backend.cast(y_true * (1 - y_pred), tensorflow.float32), axis=0)
    precision = tp / (tp + fp + keras.backend.epsilon())
    recall = tp / (tp + fn + keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + keras.backend.epsilon())
    f1 = tensorflow.where(tensorflow.is_nan(f1), tensorflow.zeros_like(f1), f1)
    micro_f1 = tensorflow.reduce_mean(f1)
    return micro_f1


def micro_f1_loss(y_true, y_pred):
    return 1 - micro_f1(y_true, y_pred)


def own_set_seed():
    # The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
    np.random.seed(0)

    # The below is necessary for starting core Python generated random numbers in a well-defined state.
    random.seed(0)

    # Force TensorFlow to use single thread. Multiple threads are a potential source of non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/
    session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    # The below tf.set_random_seed() will make random number generation in the TensorFlow backend have a well-defined
    # initial state. For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tensorflow.set_random_seed(0)
    sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)


# Returns a list with FOLDS_CV rows
# Each row has a 4-tuple (train_xs, train_ys, validations_xs, validation_ys)
# Each row represents a fold
def k_fold_cross_validation(data_xs, data_ys):
    # The folds are made by preserving the percentage of samples for each class.
    data = []
    skf = StratifiedKFold(FOLDS_CV, False, 1)

    # We need convert the data to numpy.array for indexing by other numpy.array
    data_xs = np.array(data_xs)
    data_ys = np.array(data_ys)
    for train_index, val_index in skf.split(data_xs, data_ys):
        train_xs, validation_xs = data_xs[train_index], data_xs[val_index]
        train_ys, validation_ys = data_ys[train_index], data_ys[val_index]

        data.append((train_xs, train_ys, validation_xs, validation_ys))

    return data
