from src.classifiers.adadelta_rnn import adadelta_rnn
from src.classifiers.calculated_embeddings_rnn import calculated_embeddings_rnn
from src.classifiers.pretrain_embeddings_LSTM_CONV import pretrain_embeddings_LSTM_CONV
from src.classifiers.pretrain_embeddings_rnn import pretrain_embeddings_rnn
from src.classifiers.sigmoid_pretrain_embeddings_rnn import sigmoid_pretrain_embeddings_rnn
from src.classifiers.stacked_lstm_rnn import stacked_lstm_rnn
from src.classifiers.tfidf_rnn import tfidf_rnn
from src.util.utilities import *


def main():
    embeddings_file_path = "../data/embeddings/fasttext_spanish_twitter_100d.vec"
    train_raw_tweets = get_raw_tweets('../data/input/train.xml')
    test_raw_tweets = get_raw_tweets('../data/input/test.xml')
    validation_raw_tweets = get_raw_tweets('../data/input/validation.xml')

    """ Mapping classes to numbers and vice versa """

    raw_classes = [t.find('sentiment').find('polarity').find('value').text for t in train_raw_tweets]
    CLASSES = set(raw_classes)
    NUM_TO_CLASSES_DIC = dict(enumerate(CLASSES))
    CLASSES_TO_NUM_DIC = {v: k for k, v in NUM_TO_CLASSES_DIC.items()}

    """ Reading and Loading data """

    train_tweets = [Tweet(t.find('tweetid').text, t.find('content').text,
                          CLASSES_TO_NUM_DIC[t.find('sentiment').find('polarity').find('value').text])
                    for t in train_raw_tweets]

    validation_tweets = [Tweet(t.find('tweetid').text, t.find('content').text,
                               CLASSES_TO_NUM_DIC[t.find('sentiment').find('polarity').find('value').text])
                         for t in validation_raw_tweets]

    test_tweets = [Tweet(t.find('tweetid').text, t.find('content').text) for t in test_raw_tweets]

    train_xs = get_xs(train_tweets)
    train_ys = get_ys(train_tweets)

    validation_xs = get_xs(validation_tweets)
    validation_ys = get_ys(validation_tweets)

    test_ids = get_ids(test_tweets)
    test_xs = get_xs(test_tweets)

    train_xs = preprocess_tweets(train_xs)
    validation_xs = preprocess_tweets(validation_xs)
    test_xs = preprocess_tweets(test_xs)

    """ Modelling, Evaluation and Submission """

    should_compute = {
        "tfidf_rnn": False,
        "calculated_embeddings_rnn": False,
        "ys_pretrain_embeddings_rnn": False,
        "sigmoid_pretrain_embeddings_rnn": False,
        "epochs100_pretrain_embeddings_rnn": False,
        "stacked_lstm_rnn": False,
        "adadelta_rnn": False,
        "adam_lr_0005_rnn": False,
        "pretrain_embeddings_LSTM_CONV": False,
        "preprocess_tfidf_rnn": False,
        "preprocess_calculated_embeddings_rnn": False,
        "preprocess_pretrain_embeddings_rnn": False
    }

    should_submit = {
        "tfidf_rnn": False,
        "calculated_embeddings_rnn": False,
        "ys_pretrain_embeddings_rnn": False,
        "sigmoid_pretrain_embeddings_rnn": False,
        "epochs100_pretrain_embeddings_rnn": False,
        "stacked_lstm_rnn": False,
        "adadelta_rnn": False,
        "adam_lr_0005_rnn": False,
        "pretrain_embeddings_LSTM_CONV": False
    }

    final_results = pd.DataFrame

    if should_compute["tfidf_rnn"]:
        ys_tfidf_rnn = tfidf_rnn(train_xs, train_ys, validation_xs, validation_ys)

        tfidf_rnn_results = evaluate(validation_ys, ys_tfidf_rnn, 'rnn-tfidf',
                                     classes_index=list(CLASSES_TO_NUM_DIC.values()))

        final_results = pd.concat([tfidf_rnn_results])

    if should_compute["calculated_embeddings_rnn"]:
        ys_calculated_embeddings_rnn = calculated_embeddings_rnn(embeddings_file_path, train_xs, train_ys,
                                                                 validation_xs, validation_ys)

        calculated_embeddings_rnn_results = evaluate(validation_ys, ys_calculated_embeddings_rnn,
                                                     'calculated_embeddings_rnn',
                                                     classes_index=list(CLASSES_TO_NUM_DIC.values()))
        final_results = pd.concat([calculated_embeddings_rnn_results])

    """
     A set of embeddings "represents" the vocabulary associated with the language with which we are working.
     If we try to build such vocabulary only with the words present in the training set,
     This will be very poor and may not fit the reality.
    """

    """
    ************************************* 
    Preprocesamiento de datos
    *************************************
    """

    preprocessed_train_xs = preprocess_tweets(train_xs)
    preprocessed_validation_xs = preprocess_tweets(validation_xs)
    preprocessed_test_xs = preprocess_tweets(test_xs)

    """
    ************************************* 
    Modelos 
    *************************************
    """

    if should_compute["ys_pretrain_embeddings_rnn"]:
        ys_pretrain_embeddings_rnn = pretrain_embeddings_rnn(train_xs, train_ys, validation_xs, validation_ys,
                                                             embeddings_file_path)

        pretrain_embeddings_rnn_results = evaluate(validation_ys, ys_pretrain_embeddings_rnn,
                                                   'pretrain-embeddings-rnn',
                                                   classes_index=list(CLASSES_TO_NUM_DIC.values()))
        final_results = pd.concat([pretrain_embeddings_rnn_results])

    if should_compute["sigmoid_pretrain_embeddings_rnn"]:
        ys_sigmoid_pretrain_embeddings_rnn = sigmoid_pretrain_embeddings_rnn(embeddings_file_path, train_xs, train_ys,
                                                                             validation_xs, validation_ys)

        sigmoid_pretrain_embeddings_rnn_results = evaluate(validation_ys, ys_sigmoid_pretrain_embeddings_rnn,
                                                           'sigmoid-pretrain-embeddings-rnn',
                                                           classes_index=list(CLASSES_TO_NUM_DIC.values()))
        final_results = pd.concat([sigmoid_pretrain_embeddings_rnn_results])

    if should_compute["epochs100_pretrain_embeddings_rnn"]:
        ys_epochs100_pretrain_embeddings_rnn = pretrain_embeddings_rnn(embeddings_file_path, train_xs, train_ys,
                                                                       validation_xs, validation_ys, epochs=100)

        epochs100_pretrain_embeddings_rnn_results = evaluate(validation_ys, ys_epochs100_pretrain_embeddings_rnn,
                                                             'epochs100-pretrain-embeddings-rnn',
                                                             classes_index=list(CLASSES_TO_NUM_DIC.values()))
        final_results = pd.concat([epochs100_pretrain_embeddings_rnn_results])

    if should_compute["stacked_lstm_rnn"]:
        ys_stacked_lstm_rnn = stacked_lstm_rnn(embeddings_file_path, train_xs, train_ys, validation_xs, validation_ys)

        stacked_lstm_rnn_results = evaluate(validation_ys, ys_stacked_lstm_rnn, 'stacked_lstm_rnn',
                                            classes_index=list(CLASSES_TO_NUM_DIC.values()))
        final_results = pd.concat([stacked_lstm_rnn_results])

    if should_compute["adadelta_rnn"]:
        ys_adadelta_rnn = adadelta_rnn(embeddings_file_path, train_xs, train_ys, validation_xs, validation_ys)
        adadelta_rnn_results = evaluate(validation_ys, ys_adadelta_rnn, 'adadelta_rnn',
                                        classes_index=list(CLASSES_TO_NUM_DIC.values()))
        final_results = pd.concat([adadelta_rnn_results])

    if should_compute["adam_lr_0005_rnn"]:
        ys_adam_lr_0005_rnn = pretrain_embeddings_rnn(embeddings_file_path, train_xs, train_ys, validation_xs,
                                                      validation_ys, learning_rate=0.0005)

        adam_lr_0005_results = evaluate(validation_ys, ys_adam_lr_0005_rnn, 'adam_lr_0005',
                                        classes_index=list(CLASSES_TO_NUM_DIC.values()))
        final_results = pd.concat([adam_lr_0005_results])

    if should_compute["pretrain_embeddings_LSTM_CONV"]:
        ys_pretrain_embeddings_LSTM_CONV = pretrain_embeddings_LSTM_CONV(embeddings_file_path, train_xs, train_ys,
                                                                         validation_xs, validation_ys)
        pretrain_embeddings_LSTM_CONV_results = evaluate(validation_ys, ys_pretrain_embeddings_LSTM_CONV,
                                                         'pretrain_embeddings_LSTM_CONV',
                                                         classes_index=list(CLASSES_TO_NUM_DIC.values()))
        final_results = pd.concat([pretrain_embeddings_LSTM_CONV_results])

    if should_compute["preprocess_tfidf_rnn"]:
        ys_preprocess_tfidf_rnn = tfidf_rnn(preprocessed_train_xs, train_ys, preprocessed_validation_xs, validation_ys)
        preprocess_tfidf_rnn_results = evaluate(validation_ys, ys_preprocess_tfidf_rnn, 'preprocess_rnn-tfidf',
                                                classes_index=list(CLASSES_TO_NUM_DIC.values()))
        final_results = pd.concat([preprocess_tfidf_rnn_results])

    if should_compute["preprocess_calculated_embeddings_rnn"]:
        ys_preprocess_calculated_embeddings_rnn = calculated_embeddings_rnn(preprocessed_train_xs, train_ys,
                                                                            preprocessed_validation_xs, validation_ys)
        preprocess_calculated_embeddings_rnn_results = evaluate(validation_ys, ys_preprocess_calculated_embeddings_rnn,
                                                                'preprocess_calculated_embeddings_rnn',
                                                                classes_index=list(CLASSES_TO_NUM_DIC.values()))
        final_results = pd.concat([preprocess_calculated_embeddings_rnn_results])

    if should_compute["preprocess_pretrain_embeddings_rnn"]:
        ys_preprocess_pretrain_embeddings_rnn = pretrain_embeddings_rnn(embeddings_file_path, preprocessed_train_xs,
                                                                        train_ys, preprocessed_validation_xs,
                                                                        validation_ys)
        preprocess_pretrain_embeddings_rnn_results = evaluate(validation_ys, ys_preprocess_pretrain_embeddings_rnn,
                                                              'preprocess_pretrain-embeddings-rnn',
                                                              classes_index=list(CLASSES_TO_NUM_DIC.values()))
        final_results = pd.concat([preprocess_pretrain_embeddings_rnn_results])

    """
    Why have we taken tanh instead of sigmoid? LSTMs manage an internal state vector whose values should be 
    able to increase or decrease when we add the output of some function. Sigmoid output is always non-negative; 
    values in the state would only increase. The output from tanh can be positive or negative, allowing for increases 
    and decreases in the state. 
    """
    final_results.sort_values('micro_f1', ascending=False)

    """ Submissions """

    if should_submit["tfidf_rnn"]:
        test_ys_tfidf_rnn = tfidf_rnn(train_xs, train_ys, test_xs, verbose=0)

        kaggle_file(test_ids, test_ys_tfidf_rnn, 'rnn-tfidf', NUM_TO_CLASSES_DIC)

    if should_submit["calculated_embeddings_rnn"]:
        test_ys_calculated_embeddings_rnn = calculated_embeddings_rnn(train_xs, train_ys, test_xs, verbose=0)

        kaggle_file(test_ids, test_ys_calculated_embeddings_rnn, 'calculated_embeddings_rnn', NUM_TO_CLASSES_DIC)

    if should_submit["ys_pretrain_embeddings_rnn"]:
        test_ys_pretrain_embeddings_rnn = pretrain_embeddings_rnn(embeddings_file_path,
                                                                  train_xs, train_ys, test_xs, verbose=0)
        kaggle_file(test_ids, test_ys_pretrain_embeddings_rnn, 'pretrain-embeddings-rnn', NUM_TO_CLASSES_DIC)

    if should_submit["sigmoid_pretrain_embeddings_rnn"]:
        test_ys_sigmoid_pretrain_embeddings_rnn = pretrain_embeddings_rnn(
            embeddings_file_path, train_xs, train_ys, test_xs, verbose=0)

        kaggle_file(test_ids, test_ys_sigmoid_pretrain_embeddings_rnn, 'sigmoid-pretrain-embeddings-rnn',
                    NUM_TO_CLASSES_DIC)

    if should_submit["stacked_lstm_rnn"]:
        test_ys_stacked_lstm_rnn = stacked_lstm_rnn(embeddings_file_path, train_xs, train_ys, test_xs, verbose=0)
        kaggle_file(test_ids, test_ys_stacked_lstm_rnn, 'stacked_lstm_rnn', NUM_TO_CLASSES_DIC)

    if should_submit["adam_lr_0005_rnn"]:
        test_ys_adam_lr_0005_rnn = pretrain_embeddings_rnn(embeddings_file_path,
                                                           train_xs, train_ys, test_xs, learning_rate=0.0005, verbose=0)
        kaggle_file(test_ids, test_ys_adam_lr_0005_rnn, 'adam_lr_0005', NUM_TO_CLASSES_DIC)

    if should_submit["pretrain_embeddings_LSTM_CONV"]:
        test_ys_pretrain_embeddings_LSTM_CONV = pretrain_embeddings_LSTM_CONV(
            embeddings_file_path, train_xs, train_ys, test_xs, verbose=0)

        kaggle_file(test_ids, test_ys_pretrain_embeddings_LSTM_CONV, 'pretrain_embeddings_LSTM_CONV',
                    NUM_TO_CLASSES_DIC)


if __name__ == "__main__":
    main()
