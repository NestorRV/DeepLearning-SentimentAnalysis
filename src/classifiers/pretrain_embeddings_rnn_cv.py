from src.classifiers.pretrain_embeddings_rnn import pretrain_embeddings_rnn

from src.util.utilities import *


def calculated_embeddings_rnn_cv(embeddings_path, train_xs, train_ys, test_xs, classes_to_num_dic, test_ys=None, epochs=25, verbose=1, num_classes=4):
    own_set_seed()

    model_name = "pretrain_embeddings_rnn"

    new_train_xs = train_xs+test_xs
    new_train_ys = train_ys+test_ys

    df_metrics = pd.DataFrame()

    data_k_fold = k_fold_cross_validation(new_train_xs, new_train_ys)
    for train_xs, train_ys, test_xs, test_ys in data_k_fold:
        labels_fold_i = pretrain_embeddings_rnn(embeddings_path, train_xs, train_ys, test_xs, test_ys, epochs, verbose, num_classes)
        metrics_i = evaluate(test_ys, labels_fold_i, model_name, list(classes_to_num_dic.values()))

        df_metrics = df_metrics.append(metrics_i, ignore_index=True)

    return pd.DataFrame({model_name : df_metrics.mean(axis = 0)}).T