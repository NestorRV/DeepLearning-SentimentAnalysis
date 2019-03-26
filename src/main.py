from src.classifiers.adadelta_rnn import *
from src.classifiers.calculated_embedings_rnn import *
from src.classifiers.pretrain_embedings_LSTM_CONV import *
from src.classifiers.pretrain_embedings_rnn import *
from src.classifiers.sigmoid_pretrain_embedings_rnn import *
from src.classifiers.stacked_lstm_rnn import *
from src.classifiers.tfidf_rnn import *

from src.util.utilities import *


def main():
    train_raw_tweets = get_raw_tweets('data/input/train.xml')
    test_raw_tweets = get_raw_tweets('data/input/test.xml')
    validation_raw_tweets = get_raw_tweets('data/input/validation.xml')

    """
    ************************************* 
    Mapeo de clases a un entero 
    *************************************
    """

    raw_classes = [t.find('sentiment').find('polarity').find('value').text for t in train_raw_tweets]
    CLASSES = set(raw_classes)
    NUM_TO_CLASSES_DIC = dict(enumerate(CLASSES))
    CLASSES_TO_NUM_DIC = {v: k for k, v in NUM_TO_CLASSES_DIC.items()}

    """
    ************************************* 
    Lectura de datos
    *************************************
    """

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

    """ tfidf_rnn"""
    ys_tfidf_rnn = tfidf_rnn(train_xs, train_ys, validation_xs, validation_ys)

    """ calculated_embedings_rnn"""
    ys_calculated_embeddings_rnn = calculated_embedings_rnn(train_xs, train_ys, validation_xs, validation_ys)

    """ pretrain_embedings_rnn"""
    ys_pretrain_embeddings_rnn = pretrain_embedings_rnn('data/embeddings/fasttext_spanish_twitter_100d.vec',
                                                        train_xs,
                                                        train_ys,
                                                        validation_xs, validation_ys)

    """ sigmoid_pretrain_embedings_rnn"""
    ys_sigmoid_pretrain_embeddings_rnn = sigmoid_pretrain_embedings_rnn(
        '../data/embeddings/fasttext_spanish_twitter_100d.vec', train_xs, train_ys,
        validation_xs, validation_ys)

    """ epochs100_pretrain_embeddings_rnn"""
    ys_epochs100_pretrain_embeddings_rnn = pretrain_embedings_rnn(
        '../data/embeddings/fasttext_spanish_twitter_100d.vec',
        train_xs, train_ys,
        validation_xs, validation_ys, epochs=100)

    """ stacked_lstm_rnn"""
    ys_stacked_lstm_rnn = stacked_lstm_rnn('data/embeddings/fasttext_spanish_twitter_100d.vec', train_xs, train_ys,
                                           validation_xs, validation_ys)

    """ adadelta_rnn"""
    ys_adadelta_rnn = adadelta_rnn('data/embeddings/fasttext_spanish_twitter_100d.vec', train_xs, train_ys,
                                   validation_xs, validation_ys)

    """ adam_lr_0005_rnn"""
    ys_adam_lr_0005_rnn = pretrain_embedings_rnn('data/embeddings/fasttext_spanish_twitter_100d.vec', train_xs,
                                                 train_ys,
                                                 validation_xs, validation_ys, learning_rate=0.0005)

    """ pretrain_embedings_LSTM_CONV"""
    ys_pretrain_embedings_LSTM_CONV = pretrain_embedings_LSTM_CONV(
        'data/embeddings/fasttext_spanish_twitter_100d.vec',
        train_xs, train_ys, validation_xs, validation_ys)

    """ preprocess_tfidf_rnn"""
    ys_preprocess_tfidf_rnn = tfidf_rnn(preprocessed_train_xs, train_ys, preprocessed_validation_xs, validation_ys)

    """ preprocess_calculated_embedings_rnn"""
    ys_preprocess_calculated_embeddings_rnn = calculated_embedings_rnn(preprocessed_train_xs, train_ys,
                                                                       preprocessed_validation_xs, validation_ys)

    """ preprocess_pretrain_embedings_rnn"""
    ys_preprocess_pretrain_embeddings_rnn = pretrain_embedings_rnn(
        'data/embeddings/fasttext_spanish_twitter_100d.vec',
        preprocessed_train_xs,
        train_ys,
        preprocessed_validation_xs, validation_ys)

    """
    ************************************* 
    Evaluaciones 
    *************************************
    """

    """ rnn-tfidf """
    tfidf_rnn_results = evaluate(validation_ys, ys_tfidf_rnn, 'rnn-tfidf',
                                 classes_index=list(CLASSES_TO_NUM_DIC.values()))

    """ calculated_embedings_rnn """
    calculated_embedings_rnn_results = evaluate(validation_ys, ys_calculated_embeddings_rnn, 'calculated_embedings_rnn',
                                                classes_index=list(CLASSES_TO_NUM_DIC.values()))

    # Un conjunto de embeddings "representa" el vocabulario asociado al idioma con el que estamos trabajando.
    # Si intentamos construir dicho vocabulario sólo con las palabras presentes en el conjunto de entrenamiento,
    # este será muy pobre y puede que no se adecue a la realidad.
    """ pretrain-embedings-rnn """
    pretrain_embedings_rnn_results = evaluate(validation_ys, ys_pretrain_embeddings_rnn, 'pretrain-embedings-rnn',
                                              classes_index=list(CLASSES_TO_NUM_DIC.values()))

    """ sigmoid-pretrain-embedings-rnn """
    sigmoid_pretrain_embedings_rnn_results = evaluate(validation_ys, ys_sigmoid_pretrain_embeddings_rnn,
                                                      'sigmoid-pretrain-embedings-rnn',
                                                      classes_index=list(CLASSES_TO_NUM_DIC.values()))

    """ epochs100-pretrain-embedings-rnn """
    epochs100_pretrain_embedings_rnn_results = evaluate(validation_ys, ys_epochs100_pretrain_embeddings_rnn,
                                                        'epochs100-pretrain-embedings-rnn',
                                                        classes_index=list(CLASSES_TO_NUM_DIC.values()))

    """ stacked_lstm_rnn """
    stacked_lstm_rnn_results = evaluate(validation_ys, ys_stacked_lstm_rnn, 'stacked_lstm_rnn',
                                        classes_index=list(CLASSES_TO_NUM_DIC.values()))

    """ adadelta_rnn """
    adadelta_rnn_results = evaluate(validation_ys, ys_adadelta_rnn, 'adadelta_rnn',
                                    classes_index=list(CLASSES_TO_NUM_DIC.values()))

    """ adam_lr_0005 """
    adam_lr_0005_results = evaluate(validation_ys, ys_adam_lr_0005_rnn, 'adam_lr_0005',
                                    classes_index=list(CLASSES_TO_NUM_DIC.values()))

    """ pretrain_embedings_LSTM_CONV """
    pretrain_embedings_LSTM_CONV_results = evaluate(validation_ys, ys_pretrain_embedings_LSTM_CONV,
                                                    'pretrain_embedings_LSTM_CONV_lr_0_0005',
                                                    classes_index=list(CLASSES_TO_NUM_DIC.values()))

    # Vemos que con el preprocesado se nota muchísima mejoría respecto a calculated_embeddings_rnn, ninguna mejoría con rnn_tjdif
    # y algo de mejor en pretrain_embedings
    """ preprocess_rnn-tfidf """
    preprocess_tfidf_rnn_results = evaluate(validation_ys, ys_preprocess_tfidf_rnn, 'preprocess_rnn-tfidf',
                                 classes_index=list(CLASSES_TO_NUM_DIC.values()))

    """ preprocess_calculated_embedings_rnn """
    preprocess_calculated_embedings_rnn_results = evaluate(validation_ys, ys_preprocess_calculated_embeddings_rnn, 'preprocess_calculated_embedings_rnn',
                                                classes_index=list(CLASSES_TO_NUM_DIC.values()))

    """ pretrain-embedings-rnn """
    preprocess_pretrain_embedings_rnn_results = evaluate(validation_ys, ys_preprocess_pretrain_embeddings_rnn, 'preprocess_pretrain-embedings-rnn',
                                              classes_index=list(CLASSES_TO_NUM_DIC.values()))

    # Porque hemos cogido tanh en vez de sigmoid**: LSTMs manage an internal state vector whose values should be able to
    # increase or decrease when we add the output of some function. Sigmoid output is always non-negative; values in the
    # state would only increase. The output from tanh can be positive or negative, allowing for increases and decreases in the state.

    final_results = pd.concat([tfidf_rnn_results,
                               calculated_embedings_rnn_results,
                               pretrain_embedings_rnn_results,
                               sigmoid_pretrain_embedings_rnn_results,
                               epochs100_pretrain_embedings_rnn_results,
                               stacked_lstm_rnn_results,
                               adadelta_rnn_results,
                               adam_lr_0005_results,
                               pretrain_embedings_LSTM_CONV_results,
                               preprocess_tfidf_rnn_results,
                               preprocess_calculated_embedings_rnn_results,
                               preprocess_pretrain_embedings_rnn_results])

    final_results.sort_values('micro_f1', ascending=False)

    """
    ************************************* 
    Entregas 
    *************************************
    """

    # rnn-tfidf -> Kaggle: 0.42179
    test_ys_tfidf_rnn = tfidf_rnn(train_xs, train_ys, test_xs, verbose=0)
    kaggle_file(test_ids, test_ys_tfidf_rnn, 'rnn-tfidf', NUM_TO_CLASSES_DIC=NUM_TO_CLASSES_DIC)

    # calculated_embedings_rnn -> Kaggle: 0.34622
    test_ys_calculated_embeddings_rnn = calculated_embedings_rnn(train_xs, train_ys, test_xs, verbose=0)
    kaggle_file(test_ids, test_ys_calculated_embeddings_rnn, 'calculated_embedings_rnn',
                NUM_TO_CLASSES_DIC=NUM_TO_CLASSES_DIC)

    # pretrain-embedings-rnn -> Kaggle: 0.50439
    test_ys_pretrain_embeddings_rnn = pretrain_embedings_rnn('data/embeddings/fasttext_spanish_twitter_100d.vec',
                                                             train_xs, train_ys, test_xs, verbose=0)
    kaggle_file(test_ids, test_ys_pretrain_embeddings_rnn, 'pretrain-embedings-rnn',
                NUM_TO_CLASSES_DIC=NUM_TO_CLASSES_DIC)

    # sigmoid_pretrain_embedings_rnn_results -> Kaggle: 0.50439
    test_ys_sigmoid_pretrain_embeddings_rnn = pretrain_embedings_rnn(
        'data/embeddings/fasttext_spanish_twitter_100d.vec', train_xs, train_ys, test_xs, verbose=0)
    kaggle_file(test_ids, test_ys_sigmoid_pretrain_embeddings_rnn, 'sigmoid-pretrain-embedings-rnn',
                NUM_TO_CLASSES_DIC=NUM_TO_CLASSES_DIC)

    # stacked_lstm_rnn -> Kaggle: 0.52724
    test_ys_stacked_lstm_rnn = stacked_lstm_rnn('data/embeddings/fasttext_spanish_twitter_100d.vec',
                                                train_xs, train_ys, test_xs, verbose=0)
    kaggle_file(test_ids, test_ys_stacked_lstm_rnn, 'stacked_lstm_rnn', NUM_TO_CLASSES_DIC=NUM_TO_CLASSES_DIC)

    # adam_lr_0005 -> Kaggle: 0.52899
    test_ys_adam_lr_0005_rnn = pretrain_embedings_rnn('data/embeddings/fasttext_spanish_twitter_100d.vec',
                                                      train_xs, train_ys, test_xs, learning_rate=0.0005, verbose=0)
    kaggle_file(test_ids, test_ys_adam_lr_0005_rnn, 'adam_lr_0005', NUM_TO_CLASSES_DIC=NUM_TO_CLASSES_DIC)

    # pretrain_embedings_LSTM_CONV -> Kaggle: 0.55008
    test_ys_pretrain_embedings_LSTM_CONV = pretrain_embedings_LSTM_CONV(
        'data/embeddings/fasttext_spanish_twitter_100d.vec', train_xs, train_ys, test_xs, verbose=0)
    kaggle_file(test_ids, test_ys_pretrain_embedings_LSTM_CONV, 'pretrain_embedings_LSTM_CONV',
                NUM_TO_CLASSES_DIC=NUM_TO_CLASSES_DIC)

    # preprocess_rnn-tfidf
    test_ys_preprocess_tfidf_rnn = tfidf_rnn(preprocessed_train_xs, train_ys, preprocessed_test_xs, verbose=0)
    kaggle_file(test_ids, test_ys_preprocess_tfidf_rnn, 'preprocess_rnn-tfidf', NUM_TO_CLASSES_DIC=NUM_TO_CLASSES_DIC)

    # preprocess_calculated_embedings_rnn
    test_ys_preprocess_calculated_embeddings_rnn = calculated_embedings_rnn(preprocessed_train_xs, train_ys, preprocessed_test_xs, verbose=0)
    kaggle_file(test_ids, test_ys_preprocess_calculated_embeddings_rnn, 'preprocess_calculated_embedings_rnn',
                NUM_TO_CLASSES_DIC=NUM_TO_CLASSES_DIC)

    # preprocess_pretrain-embedings-rnn
    test_ys_preprocess_pretrain_embeddings_rnn = pretrain_embedings_rnn('data/embeddings/fasttext_spanish_twitter_100d.vec',
                                                             preprocessed_train_xs, train_ys, preprocessed_test_xs, verbose=0)
    kaggle_file(test_ids, test_ys_preprocess_pretrain_embeddings_rnn, 'preprocess_pretrain-embedings-rnn',
                NUM_TO_CLASSES_DIC=NUM_TO_CLASSES_DIC)


if __name__ == "__main__":
    main()
