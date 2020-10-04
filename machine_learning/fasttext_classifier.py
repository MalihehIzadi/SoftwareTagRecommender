from functools import partial
from pprint import pprint

import click
import fasttext
import numpy as np
import pandas as pd
import sklearn
from scipy import stats


@click.command()
@click.option('--train', prompt='train CSV file path', help='train CSV file path.')
@click.option('--test', prompt='test CSV file path', help='test CSV file path.')
@click.option('--topics_column', prompt='Topics Column name', help='The name of topics column.')
@click.option('--readme_column', prompt='Text Column name', help='The name of readme text column.')
@click.option('--model_output', default='fasttext_model', help='Model save path.')
@click.option('--learning_rate', default=0.05, help='Learning rate Value.')
@click.option('--epoch', default=100, help='Number of Epoch.')
@click.option('--word_ngrams', default=2, help='Number of wordNgrams.')
def ft(train, test, topics_column, readme_column, model_output, learning_rate, epoch, word_ngrams):
    """Train and Evaluate data on fastText model"""
    train = pd.read_csv(train)
    test = pd.read_csv(test)

    def make_fasttext_train(d, file):
        __ = "\n"
        with open(file, "w") as file:
            for _, i in d.iterrows():
                res = ""
                for j in str(i[topics_column]).split(','):
                    res += f'__label__{j} '

                res += f'{str(i[readme_column]).replace(__, " ")}'
                file.write(res + '\n')

    make_fasttext_train(train, 'train.txt')
    make_fasttext_train(test, 'test.txt')

    model = fasttext.train_supervised(input="train.txt", lr=learning_rate, epoch=epoch, wordNgrams=word_ngrams)

    train[topics_column] = train[topics_column].astype(str)
    _ = list(train[topics_column].map(lambda x: x.split(',')))
    _ = [i for s in _ for i in s]
    topics_list = list(set(_))

    test[topics_column] = test[topics_column].astype(str)

    def get_binary_list(topics):
        l = [0] * len(topics_list)
        for topic in topics.split(','):
            l[topics_list.index(topic)] = 1
        return l

    y_pred = []
    y_original = []
    for i in test[topics_column]:
        y_original.append(get_binary_list(i))

    for i in test[readme_column]:
        x = model.predict(i, k=-1, threshold=0)
        l = [0] * len(topics_list)
        for j, k in zip(x[0], x[1]):
            l[topics_list.index(j.replace("__label__", ""))] = k
        y_pred.append(l)

    num_selected_labels = len(topics_list)

    def eval(text, ts):
        def calc(p1, p2, func, **kwargs):
            p2 = [list(map(lambda x: 1 if x > 0.5 else 0, y)) for y in p2]
            return func(p1, p2, **kwargs)

        def calc_recom(p1, p2, func, **kwargs):
            return func(p1, p2, **kwargs)

        def success_rate(y_original, y_pred):
            common = 0
            for i in range(0, len(y_pred)):
                if (sum(np.array(y_original[i]) * np.array(y_pred[i]))) > 0:
                    common = common + 1
            success = common / len(y_pred)
            return success

        def coverage(y_original, y_pred):
            x = np.sum(y_pred, axis=0)
            c = np.count_nonzero(x > 0)
            cov = c / num_selected_labels
            return cov

        def prf_at_k(y_original, y_pred_probab):
            org_label_count_vec = np.sum(y_original, axis=1)
            repo_2_tags = len(np.where(org_label_count_vec >= 2)[0])
            repo_5_tags = len(np.where(org_label_count_vec >= 5)[0])
            k_list = [1, 2, 3, 5, 8, 10]
            s1, s5 = {}, {}
            r, p, f = {}, {}, {}
            y_org_array = np.array(y_original)
            for k in k_list:
                org_label_count = np.sum(y_org_array, axis=1).tolist()
                top_ind = []
                top_ind = np.argpartition(y_pred_probab, -1 * k, axis=1)[:, -1 * k:]
                pred_in_org = y_org_array[np.arange(len(y_org_array))[:, None], top_ind]
                common_topk = np.sum(pred_in_org, axis=1)

                recall, precision, f1 = [], [], []
                success1, success5 = 0, 0
                for index, value in enumerate(common_topk):
                    recall.append(value / min(k, org_label_count[index]))
                    precision.append(value / k)

                    if (value >= 1): success1 += 1
                    if (value >= 5): success5 += 1

                s1.update({'S1@' + str(k): "{:.2f}".format((success1 / len(y_original)) * 100)})
                s5.update({'S5@' + str(k): "{:.2f}".format((success5 / repo_5_tags) * 100)})
                r.update({'R@' + str(k): "{:.2f}".format(np.mean(recall) * 100)})
                p.update({'P@' + str(k): "{:.2f}".format(np.mean(precision) * 100)})
                f1 = stats.hmean([precision, recall])
                f.update({'F1@' + str(k): "{:.2f}".format(np.mean(f1) * 100)})
            return r, p, f, s1, s5

        metrics = {
            "Success_Rate": partial(calc, func=success_rate),
            "Coverage": partial(calc, func=coverage),
            "LRL": partial(calc, func=sklearn.metrics.label_ranking_loss),
            "AUC_micro": partial(calc, func=sklearn.metrics.roc_auc_score, average='micro'),
            "AUC_macro": partial(calc, func=sklearn.metrics.roc_auc_score, average='macro'),
            "AUC_wighted": partial(calc, func=sklearn.metrics.roc_auc_score, average='weighted'),
            "Coverage_err": partial(calc, func=sklearn.metrics.coverage_error),
            "Avg_P_score_micro": partial(calc, func=sklearn.metrics.average_precision_score, average='micro'),
            "Avg_P_score_macro": partial(calc, func=sklearn.metrics.average_precision_score, average='macro'),
            "f1_micro": partial(calc, func=sklearn.metrics.f1_score, average='micro'),
            "f1_macro": partial(calc, func=sklearn.metrics.f1_score, average='macro'),
            "f1_weighted": partial(calc, func=sklearn.metrics.f1_score, average='weighted'),
            "f1_samples": partial(calc, func=sklearn.metrics.f1_score, average='samples'),
            "prec_micro": partial(calc, func=sklearn.metrics.precision_score, average='micro'),
            "prec_macro": partial(calc, func=sklearn.metrics.precision_score, average='macro'),
            "prec_weighted": partial(calc, func=sklearn.metrics.precision_score, average='weighted'),
            "prec_samples": partial(calc, func=sklearn.metrics.precision_score, average='samples'),
            "recall_micro": partial(calc, func=sklearn.metrics.recall_score, average='micro'),
            "recall_macro": partial(calc, func=sklearn.metrics.recall_score, average='macro'),
            "recall_weighted": partial(calc, func=sklearn.metrics.recall_score, average='weighted'),
            "recall_samples": partial(calc, func=sklearn.metrics.recall_score, average='samples'),
            "hamming_loss": partial(calc, func=sklearn.metrics.hamming_loss),
            "exact_match_ratio": partial(calc, func=sklearn.metrics.accuracy_score),
            "R@k": partial(calc_recom, func=prf_at_k)
        }

        results = {i: metrics[i](y_original, y_pred) for i in metrics}
        return results

    pprint(eval(y_original, y_pred), indent=2)

    model.save_model(model_output)


if __name__ == "__main__":
    ft()
