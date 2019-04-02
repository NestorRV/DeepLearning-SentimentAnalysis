import src.util.global_vars
from src.classifiers.cv.adadelta_rnn_cv import adadelta_rnn_cv
from src.classifiers.cv.big_LSTM_CONV_rnn_cv import big_LSTM_CONV_rnn_cv
from src.classifiers.cv.calculated_embeddings_LSTM_CONV_cv import calculated_embeddings_LSTM_CONV_cv
from src.classifiers.cv.calculated_embeddings_rnn_cv import calculated_embeddings_rnn_cv
from src.classifiers.cv.dropout_LSTM_CONV_rnn_cv import dropout_LSMT_CONV_rnn_cv
from src.classifiers.cv.pretrain_embeddings_LSTM_CONV_cv import pretrain_embeddings_LSTM_CONV_cv
from src.classifiers.cv.pretrain_embeddings_rnn_cv import pretrain_embeddings_rnn_cv
from src.classifiers.cv.sigmoid_pretrain_embeddings_rnn_cv import sigmoid_pretrain_embeddings_rnn_cv
from src.classifiers.cv.stacked_lstm_rnn_cv import stacked_lstm_rnn_cv
from src.classifiers.cv.tfidf_rnn_cv import tfidf_rnn_cv
from src.classifiers.single.big_LSTM_CONV_rnn import big_LSTM_CONV_rnn
from src.classifiers.single.calculated_embeddings_LSTM_CONV import calculated_embeddings_LSTM_CONV
from src.classifiers.single.calculated_embeddings_rnn import calculated_embeddings_rnn
from src.classifiers.single.dropout_LSTM_CONV_rnn import dropout_LSMT_CONV_rnn
from src.classifiers.single.pretrain_embeddings_LSTM_CONV import pretrain_embeddings_LSTM_CONV
from src.classifiers.single.pretrain_embeddings_rnn import pretrain_embeddings_rnn
from src.classifiers.single.sigmoid_pretrain_embeddings_rnn import sigmoid_pretrain_embeddings_rnn
from src.classifiers.single.stacked_lstm_rnn import stacked_lstm_rnn
from src.classifiers.single.tfidf_rnn import tfidf_rnn
from src.util.utilities import *


def main():
    embeddings_file_path = "../data/embeddings/fasttext_spanish_twitter_100d.vec"
    train_raw_tweets = get_raw_tweets('../data/input/train.xml')
    test_raw_tweets = get_raw_tweets('../data/input/test.xml')
    validation_raw_tweets = get_raw_tweets('../data/input/validation.xml')

    """ Mapping classes to numbers and vice versa """

    raw_classes = [t.find('sentiment').find('polarity').find('value').text for t in train_raw_tweets]

    src.util.global_vars.__CLASSES__ = set(raw_classes)
    src.util.global_vars.__NUM_TO_CLASSES_DIC__ = dict(enumerate(src.util.global_vars.__CLASSES__))
    src.util.global_vars.__CLASSES_TO_NUM_DIC__ = {v: k for k, v in src.util.global_vars.__NUM_TO_CLASSES_DIC__.items()}

    """ Reading and Loading data """

    train_tweets = [Tweet(t.find('tweetid').text, t.find('content').text,
                          src.util.global_vars.__CLASSES_TO_NUM_DIC__[
                              t.find('sentiment').find('polarity').find('value').text])
                    for t in train_raw_tweets]

    validation_tweets = [Tweet(t.find('tweetid').text, t.find('content').text,
                               src.util.global_vars.__CLASSES_TO_NUM_DIC__[
                                   t.find('sentiment').find('polarity').find('value').text])
                         for t in validation_raw_tweets]

    test_tweets = [Tweet(t.find('tweetid').text, t.find('content').text) for t in test_raw_tweets]

    train_xs = get_xs(train_tweets)
    train_ys = get_ys(train_tweets)

    validation_xs = get_xs(validation_tweets)
    validation_ys = get_ys(validation_tweets)

    test_ids = get_ids(test_tweets)
    test_xs = get_xs(test_tweets)

    """ Modelling, Evaluation and Submission """

    should_compute = {
        "tfidf_rnn": False,
        "calculated_embeddings_rnn": False,
        "pretrain_embeddings_rnn": False,
        "sigmoid_pretrain_embeddings_rnn": False,
        "epochs100_pretrain_embeddings_rnn": False,
        "stacked_lstm_rnn": False,
        "adadelta_rnn": False,
        "adam_lr_0005_rnn": False,
        "pretrain_embeddings_LSTM_CONV": False,
        "preprocess_tfidf_rnn": False,
        "preprocess_calculated_embeddings_rnn": False,
        "preprocess_pretrain_embeddings_rnn": False,
        "big_LSTM_CONV_rnn": False,
        "dropout_LSTM_CONV_rnn": False,
        "preprocess_pretrain_embeddings_LSTM_CONV": False,
        "preprocess_calculated_embeddings_LSTM_CONV": False,
        "epochs50_preprocess_calculated_embeddings_LSTM_CONV": False,
    }

    final_results = pd.DataFrame

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

    if should_compute["tfidf_rnn"]:
        tfidf_rnn_results = tfidf_rnn_cv('tfidf_rnn', train_xs, train_ys, validation_xs, validation_ys)
        final_results = pd.concat([tfidf_rnn_results])

        test_ys_tfidf_rnn, _ = tfidf_rnn(train_xs, train_ys, test_xs, verbose=0)
        kaggle_file(test_ids, test_ys_tfidf_rnn, 'rnn-tfidf')

    if should_compute["calculated_embeddings_rnn"]:
        calculated_embeddings_rnn_results = calculated_embeddings_rnn_cv('calculated_embeddings_rnn', train_xs,
                                                                         train_ys, validation_xs, validation_ys)
        final_results = pd.concat([calculated_embeddings_rnn_results])

        test_ys_calculated_embeddings_rnn, _ = calculated_embeddings_rnn(train_xs, train_ys, test_xs, verbose=0)
        kaggle_file(test_ids, test_ys_calculated_embeddings_rnn, 'calculated_embeddings_rnn')

    if should_compute["pretrain_embeddings_rnn"]:
        ys_pretrain_embeddings_rnn_results = pretrain_embeddings_rnn_cv('pretrain_embeddings_rnn',
                                                                        embeddings_file_path, train_xs, train_ys,
                                                                        validation_xs, validation_ys)
        final_results = pd.concat([ys_pretrain_embeddings_rnn_results])

        test_ys_pretrain_embeddings_rnn, _ = pretrain_embeddings_rnn(embeddings_file_path, train_xs, train_ys, test_xs,
                                                                     verbose=0)
        kaggle_file(test_ids, test_ys_pretrain_embeddings_rnn, 'pretrain-embeddings-rnn')

    if should_compute["sigmoid_pretrain_embeddings_rnn"]:
        sigmoid_pretrain_embeddings_rnn_results = sigmoid_pretrain_embeddings_rnn_cv('sigmoid_pretrain_embeddings_rnn',
                                                                                     embeddings_file_path, train_xs,
                                                                                     train_ys, validation_xs,
                                                                                     validation_ys)
        final_results = pd.concat([sigmoid_pretrain_embeddings_rnn_results])

        test_ys_sigmoid_pretrain_embeddings_rnn, _ = sigmoid_pretrain_embeddings_rnn(embeddings_file_path, train_xs,
                                                                                     train_ys, test_xs, verbose=0)
        kaggle_file(test_ids, test_ys_sigmoid_pretrain_embeddings_rnn, 'sigmoid-pretrain-embeddings-rnn')

    if should_compute["epochs100_pretrain_embeddings_rnn"]:
        epochs100_pretrain_embeddings_rnn_results = pretrain_embeddings_rnn_cv('epochs100_pretrain_embeddings_rnn',
                                                                               embeddings_file_path, train_xs, train_ys,
                                                                               validation_xs, validation_ys, epochs=100)
        final_results = pd.concat([epochs100_pretrain_embeddings_rnn_results])

        test_ys_epochs100_pretrain_embeddings_rnn, _ = pretrain_embeddings_rnn(embeddings_file_path, train_xs, train_ys,
                                                                               test_xs, epochs=100)
        kaggle_file(test_ids, test_ys_epochs100_pretrain_embeddings_rnn,
                    'sigmoid-epochs100_pretrain_embeddings_rnn-embeddings-rnn')

    if should_compute["stacked_lstm_rnn"]:
        stacked_lstm_rnn_results = stacked_lstm_rnn_cv('stacked_lstm_rnn', embeddings_file_path, train_xs, train_ys,
                                                       validation_xs, validation_ys)
        final_results = pd.concat([stacked_lstm_rnn_results])

        test_ys_stacked_lstm_rnn, _ = stacked_lstm_rnn(embeddings_file_path, train_xs, train_ys, test_xs, verbose=0)
        kaggle_file(test_ids, test_ys_stacked_lstm_rnn, 'stacked_lstm_rnn')

    if should_compute["adadelta_rnn"]:
        adadelta_rnn_results = adadelta_rnn_cv('adadelta_rnn', embeddings_file_path, train_xs, train_ys, validation_xs,
                                               validation_ys)
        final_results = pd.concat([adadelta_rnn_results])

        test_ys_adadelta_rnn, _ = stacked_lstm_rnn(embeddings_file_path, train_xs, train_ys, test_xs, verbose=0)
        kaggle_file(test_ids, test_ys_adadelta_rnn, 'adadelta_rnn')

    if should_compute["adam_lr_0005_rnn"]:
        adam_lr_0005_rnn_results = pretrain_embeddings_rnn_cv('adam_lr_0005_rnn', embeddings_file_path, train_xs,
                                                              train_ys, validation_xs, validation_ys,
                                                              learning_rate=0.0005)
        final_results = pd.concat([adam_lr_0005_rnn_results])

        test_ys_adam_lr_0005_rnn, _ = pretrain_embeddings_rnn(embeddings_file_path, train_xs, train_ys, test_xs,
                                                              learning_rate=0.0005, verbose=0)
        kaggle_file(test_ids, test_ys_adam_lr_0005_rnn, 'adam_lr_0005')

    if should_compute["pretrain_embeddings_LSTM_CONV"]:
        pretrain_embeddings_LSTM_CONV_results = pretrain_embeddings_LSTM_CONV_cv('pretrain_embeddings_LSTM_CONV',
                                                                                 embeddings_file_path, train_xs,
                                                                                 train_ys, validation_xs, validation_ys)
        final_results = pd.concat([pretrain_embeddings_LSTM_CONV_results])

        test_ys_pretrain_embeddings_LSTM_CONV, _ = pretrain_embeddings_LSTM_CONV(embeddings_file_path, train_xs,
                                                                                 train_ys,
                                                                                 test_xs, verbose=0)
        kaggle_file(test_ids, test_ys_pretrain_embeddings_LSTM_CONV, 'pretrain_embeddings_LSTM_CONV')

    if should_compute["preprocess_tfidf_rnn"]:
        preprocess_tfidf_rnn_results = tfidf_rnn_cv('preprocess_tfidf_rnn', preprocessed_train_xs, train_ys,
                                                    preprocessed_validation_xs, validation_ys)
        final_results = pd.concat([preprocess_tfidf_rnn_results])

        test_ys_preprocess_tfidf_rnn, _ = tfidf_rnn(preprocessed_train_xs, train_ys, preprocessed_test_xs, verbose=0)
        kaggle_file(test_ids, test_ys_preprocess_tfidf_rnn, 'preprocess_tfidf_rnn')

    if should_compute["preprocess_calculated_embeddings_rnn"]:
        preprocess_calculated_embeddings_rnn_results = calculated_embeddings_rnn_cv(
            'preprocess_calculated_embeddings_rnn', preprocessed_train_xs, train_ys, preprocessed_validation_xs,
            validation_ys)
        final_results = pd.concat([preprocess_calculated_embeddings_rnn_results])

        test_ys_preprocess_calculated_embeddings_rnn, _ = calculated_embeddings_rnn(preprocessed_train_xs, train_ys,
                                                                                    preprocessed_test_xs, verbose=0)
        kaggle_file(test_ids, test_ys_preprocess_calculated_embeddings_rnn, 'preprocess_calculated_embeddings_rnn')

    if should_compute["preprocess_pretrain_embeddings_rnn"]:
        preprocess_pretrain_embeddings_rnn_results = pretrain_embeddings_rnn_cv('preprocess_pretrain_embeddings_rnn',
                                                                                embeddings_file_path,
                                                                                preprocessed_train_xs, train_ys,
                                                                                preprocessed_validation_xs,
                                                                                validation_ys)
        final_results = pd.concat([preprocess_pretrain_embeddings_rnn_results])

        test_ys_preprocess_pretrain_embeddings_rnn, _ = pretrain_embeddings_rnn(embeddings_file_path,
                                                                                preprocessed_train_xs, train_ys,
                                                                                preprocessed_test_xs, verbose=0)
        kaggle_file(test_ids, test_ys_preprocess_pretrain_embeddings_rnn, 'preprocess_pretrain_embeddings_rnn')

    if should_compute["preprocess_pretrain_embeddings_LSTM_CONV"]:
        preprocess_pretrain_embeddings_LSTM_CONV_results = pretrain_embeddings_LSTM_CONV_cv(
            'preprocess_pretrain_embeddings_LSTM_CONV', embeddings_file_path, preprocessed_train_xs, train_ys,
            preprocessed_validation_xs, validation_ys)
        final_results = pd.concat([preprocess_pretrain_embeddings_LSTM_CONV_results])

        test_ys_preprocess_pretrain_embeddings_LSTM_CONV, _ = pretrain_embeddings_LSTM_CONV(embeddings_file_path,
                                                                                            preprocessed_train_xs,
                                                                                            train_ys,
                                                                                            preprocessed_test_xs,
                                                                                            verbose=0)
        kaggle_file(test_ids, test_ys_preprocess_pretrain_embeddings_LSTM_CONV,
                    'preprocess_pretrain_embeddings_LSTM_CONV')

    if should_compute["preprocess_calculated_embeddings_LSTM_CONV"]:
        stemming_preprocessed_train_xs = preprocess_tweets(train_xs, True)
        stemming_preprocessed_validation_xs = preprocess_tweets(validation_xs, True)

        preprocess_calculated_embeddings_LSTM_CONV_results = calculated_embeddings_LSTM_CONV_cv(
            'preprocess_calculated_embeddings_LSTM_CONV', stemming_preprocessed_train_xs, train_ys,
            stemming_preprocessed_validation_xs, validation_ys)
        final_results = pd.concat([preprocess_calculated_embeddings_LSTM_CONV_results])

        test_ys_preprocess_calculated_embeddings_LSTM_CONV, _ = calculated_embeddings_LSTM_CONV(preprocessed_train_xs,
                                                                                                train_ys,
                                                                                                preprocessed_test_xs,
                                                                                                verbose=0)
        kaggle_file(test_ids, test_ys_preprocess_calculated_embeddings_LSTM_CONV,
                    'preprocess_calculated_embeddings_LSTM_CONV')

    if should_compute["epochs50_preprocess_calculated_embeddings_LSTM_CONV"]:
        stemming_preprocessed_train_xs = preprocess_tweets(train_xs, True)
        stemming_preprocessed_validation_xs = preprocess_tweets(validation_xs, True)

        epochs50_preprocess_calculated_embeddings_LSTM_CONV_results = calculated_embeddings_LSTM_CONV_cv(
            'epochs50_preprocess_calculated_embeddings_LSTM_CONV', stemming_preprocessed_train_xs, train_ys,
            stemming_preprocessed_validation_xs, validation_ys, epochs=50)
        final_results = pd.concat([epochs50_preprocess_calculated_embeddings_LSTM_CONV_results])

        test_ys_epochs50_preprocess_calculated_embeddings_LSTM_CONV, _ = calculated_embeddings_LSTM_CONV(
            embeddings_file_path, preprocessed_train_xs, train_ys, preprocessed_test_xs, verbose=0, epochs=50)
        kaggle_file(test_ids, test_ys_epochs50_preprocess_calculated_embeddings_LSTM_CONV,
                    'epochs50_preprocess_calculated_embeddings_LSTM_CONV')

    if should_compute["big_LSTM_CONV_rnn"]:
        big_LSTM_CONV_rnn_results = big_LSTM_CONV_rnn_cv('big_LSTM_CONV_rnn', embeddings_file_path, train_xs, train_ys,
                                                         validation_xs, validation_ys)
        final_results = pd.concat([big_LSTM_CONV_rnn_results])

        test_ys_big_LSTM_CONV_rnn, _ = big_LSTM_CONV_rnn(embeddings_file_path, preprocessed_train_xs, train_ys, test_xs,
                                                         verbose=0)
        kaggle_file(test_ids, test_ys_big_LSTM_CONV_rnn, 'big_LSTM_CONV_rnn')

    if should_compute["dropout_LSTM_CONV_rnn"]:
        dropout_LSTM_CONV_rnn_results = dropout_LSMT_CONV_rnn_cv('dropout_LSTM_CONV_rnn', embeddings_file_path,
                                                                 train_xs, train_ys, validation_xs, validation_ys)
        final_results = pd.concat([dropout_LSTM_CONV_rnn_results])

        test_ys_dropout_LSTM_CONV_rnn, _ = dropout_LSMT_CONV_rnn(embeddings_file_path, train_xs, train_ys, test_xs,
                                                                 verbose=0)
        kaggle_file(test_ids, test_ys_dropout_LSTM_CONV_rnn, 'dropout_LSTM_CONV_rnn')

    final_results.sort_values('micro_f1', ascending=False)
    print(final_results)


if __name__ == "__main__":
    main()
