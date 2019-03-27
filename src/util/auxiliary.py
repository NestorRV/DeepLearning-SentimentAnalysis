from src.classifiers.pretrained_embeddings.adadelta_rnn import adadelta_rnn
from src.classifiers.pretrained_embeddings.stacked_lstm_rnn import stacked_lstm_rnn
from src.classifiers.others.tfidf_rnn import tfidf_rnn
from classifiers.pretrained_embeddings.pretrain_embeddings_LSTM_CONV import pretrain_embeddings_LSTM_CONV
from src.classifiers.pretrained_embeddings.pretrain_embeddings_rnn import pretrain_embeddings_rnn
from src.classifiers.pretrained_embeddings.sigmoid_pretrain_embeddings_rnn import sigmoid_pretrain_embedings_rnn

def modelling(model_number, train_xs, train_ys, validation_xs, validation_ys, embedding_file="", epochs=100,
              learning_rate=0.0001):
    global model

    if model_number == 0:
        model = tfidf_rnn(train_xs, train_ys, validation_xs, validation_ys)

    elif model_number == 1:
        from src.classifiers.calculated_embeddings.calculated_embeddings_rnn import calculated_embeddings_rnn
        model = calculated_embeddings_rnn(train_xs, train_ys, validation_xs, validation_ys)

    elif model_number == 2:
        model = pretrain_embedings_rnn(embedding_file, train_ys, validation_xs, validation_ys)

    elif model_number == 3:
        model = sigmoid_pretrain_embedings_rnn(embedding_file, train_xs, train_ys, validation_xs, validation_ys)

    elif model_number == 4:
        model = pretrain_embedings_rnn(embedding_file, train_xs, train_ys, validation_xs, validation_ys, epochs)

    elif model_number == 5:
        model = stacked_lstm_rnn(embedding_file, train_xs, train_ys, validation_xs, validation_ys)

    elif model_number == 6:
        model = adadelta_rnn(embedding_file, train_xs, train_ys, validation_xs, validation_ys)

    elif model_number == 7:
        model = pretrain_embedings_rnn(embedding_file, train_xs, train_ys, validation_xs, validation_ys, learning_rate)

    elif model_number == 8:
        model = pretrain_embeddings_LSTM_CONV(embedding_file, train_xs, train_ys, validation_xs, validation_ys)

    return model