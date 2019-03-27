import re
import time
import xml.etree.ElementTree
from collections import OrderedDict

import pandas as pd
from matplotlib import pyplot
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
from num2words import num2words
from numpy import array as np_array
from sklearn import metrics
from unidecode import unidecode

TWEET_TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
EMB_SEP_CHAR = " "
RE_TOKEN_USER = re.compile(
    r"(?<![A-Za-z0-9_!@#$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)")


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


def evaluate(real_ys, predicted_ys, model_name, classes_index):
    """
    evaluate, is a function used to obtain the evaluation metrics for model_name with real_ys as the real labels and
    predicted_ys as the predicted labels.

    :param real_ys: the real labels
    :param predicted_ys: the predicted labels
    :param model_name: the model to evaluate
    :param classes_index: the corresponding index for each class label
    :return:
    """
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


def kaggle_file(ids, ys, model, num_to_classes_dic):
    """
    src.util.utilities.kaggle_file, a function used to prepare the submission file.

    :param ids: instances ID's
    :param ys: the predicted labels
    :param model: the model name
    :param num_to_classes_dic: a dictionary of index: "real class label" pairs.
    :return:
    """
    real_classes = [num_to_classes_dic[y] for y in ys]
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


def plot_graphic(history, name):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title(name)
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Validation'], loc='upper right')
    pyplot.savefig("../plots" + name + '-' + str(int(time.time())) + '.png')


def remove_emojis(tweet):
    # :), : ), :-), (:, ( :, (-:, :'), :D, : D, :-D, xD, x-D, XD, X-D, xd
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D|xd)', ' EMO_POS ', tweet)
    # jajaja, lol
    tweet = re.sub(r'(a*ja+j[ja]*|o?l+o+l+[ol]*)', ' EMO_POS ', tweet)
    # hahaha
    tweet = re.sub(r'(a*ha+h[ha])', ' EMO_POS ', tweet)
    # <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # ;-), ;), ;-D, ;D, (;,  (-;, ^^
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;|\^\^)', ' EMO_POS ', tweet)
    # :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweets(tweets):
    # download('stopwords')
    # download('punkt')
    stop_words = stopwords.words('spanish')

    preprocessed_tweets = []
    # lower case and remove accent marks
    lowercase_tweets = [unidecode(tweet.lower()) for tweet in tweets]

    for tweet in lowercase_tweets:
        # remove punctuation
        tweet = tweet.strip('\'"?!,.():;')
        # convert more than 2 letter repetitions to 2 letter
        tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
        # Remove - & '
        tweet = re.sub(r'(-|\')', '', tweet)

        # Replaces URLs with the word URL
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
        # Replace @handle with the word USER_MENTION
        tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
        # Replaces #hashtag with hashtag
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
        # Remove RT (retweet)
        tweet = re.sub(r'\brt\b', '', tweet)
        # Replace 2+ dots with space
        tweet = re.sub(r'\.{2,}', ' ', tweet)
        # Strip space, " and ' from tweet
        tweet = tweet.strip(' "\'')
        # Replace emojis with either EMO_POS or EMO_NEG
        tweet = remove_emojis(tweet)
        # Replace multiple spaces with a single space
        tweet = re.sub(r'\s+', ' ', tweet)

        tweet_words = word_tokenize(tweet)
        # number to words
        tweet_words = [num2words(float(word), lang='es') if word.isnumeric() else word for word in tweet_words]
        # remove stopwords
        tweet_words = ' '.join(word for word in tweet_words if word not in stop_words)

        preprocessed_tweets.append(tweet_words)

    return preprocessed_tweets
