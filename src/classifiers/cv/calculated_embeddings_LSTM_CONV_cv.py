from src.classifiers.single.calculated_embeddings_LSTM_CONV import calculated_embeddings_LSTM_CONV
from src.util.utilities import *


def calculated_embeddings_LSTM_CONV_cv(train_xs, train_ys, test_xs, test_ys=None, verbose=1, epochs=25):
    own_set_seed()

    model_name = "calculated_embeddings_rnn"

    new_train_xs = np.concatenate((np.array(train_xs), np.array(test_xs)))
    new_train_ys = np.concatenate((np.array(train_ys), np.array(test_ys)))

    df_metrics = pd.DataFrame()
    histories = []

    data_k_fold = k_fold_cross_validation(new_train_xs, new_train_ys)
    for train_xs, train_ys, val_xs, val_ys in data_k_fold:
        labels_fold_i, history_i = calculated_embeddings_LSTM_CONV(train_xs, train_ys, val_xs, val_ys, verbose, epochs)
        metrics_i = evaluate(val_ys, labels_fold_i, model_name)

        df_metrics = df_metrics.append(metrics_i, ignore_index=True)
        histories.append(history_i)

    plot_graphic(histories, model_name)
    return pd.DataFrame({model_name: df_metrics.mean(axis=0)}).T
