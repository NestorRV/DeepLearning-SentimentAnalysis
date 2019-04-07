from sklearn.metrics import confusion_matrix

from src.classifiers.single.bidirectional_lstm_rnn import bidirectional_lstm_rnn

from src.util.utilities import *


def bidirectional_lstm_rnn_cv(model_name, embeddings_path, train_xs, train_ys, test_xs, test_ys=None, verbose=1):
    own_set_seed()

    new_train_xs = np.concatenate((np.array(train_xs), np.array(test_xs)))
    new_train_ys = np.concatenate((np.array(train_ys), np.array(test_ys)))

    df_metrics = pd.DataFrame()
    histories = []

    data_k_fold = k_fold_cross_validation(new_train_xs, new_train_ys)
    for train_xs, train_ys, val_xs, val_ys in data_k_fold:
        labels_fold_i, history_i = bidirectional_lstm_rnn(embeddings_path, train_xs, train_ys, val_xs, val_ys, verbose)
        metrics_i = evaluate(val_ys, labels_fold_i, model_name)
        print(confusion_matrix(val_ys, labels_fold_i, labels=range(4)))

        df_metrics = df_metrics.append(metrics_i, ignore_index=True)
        histories.append(history_i)

    plot_graphic(histories, model_name)
    return pd.DataFrame({model_name: df_metrics.mean(axis=0)}).T
