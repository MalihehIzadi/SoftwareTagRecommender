import logging
import os
import pickle
import warnings
from functools import partial

import click
import gensim
import numpy as np
import pandas as pd
import scipy
import sklearn
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from skmultilearn.problem_transform import ClassifierChain
from tqdm import tqdm

warnings.filterwarnings('ignore')


@click.command()
@click.option('--train', prompt='train CSV file path', help='train CSV file path.')
@click.option('--test', prompt='test CSV file path', help='test CSV file path.')
@click.option('--topics_column', prompt='Topics Column name', help='The name of topics column.')
@click.option('--readme_column', prompt='Text Column name', help='The name of readme text column.')
@click.option('--model',
              type=click.Choice(['nb', 'gnb', 'lr', 'sgd', 'svm'], case_sensitive=False, help='Model Type.'))
@click.option('--tokenizer',
              type=click.Choice(['tfidf', 'doc2vec'], case_sensitive=False, help='Tokenizer Type.'))
@click.option('--method',
              type=click.Choice(['ovr', 'cc'], case_sensitive=False,
                                help="Methods between ClassifierChain or OneVsRest."))
def ft(train, test, topics_column, readme_column, tokenizer, model, method):
    """Train and Evaluation on Traditional ML Models."""

    topics_col = topics_column
    text_col = readme_column

    # doc2vec params
    minCount = 10
    d2v_max_feat = 1000

    # tfidf params
    ngramRange = (1, 2)
    tfidf_max_feat = 20000

    # different setting for tf-idf, doc2vc, and tuning modes
    featuresMode = ["", "doc2vec"]
    tuningModes = ["", "grid search", "randomized search"]

    if os.path.exists('models'):
        os.mkdir("models")

    models_path = 'models/'

    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)

    train_df = train_df.drop(columns=['labels'])
    test_df = test_df.drop(columns=['labels'])

    X_train = train_df[[text_col]]
    y_train = train_df[train_df.columns.difference([text_col])]

    X_test = test_df[[text_col]]
    y_test = test_df[test_df.columns.difference([text_col])]

    X_train[text_col] = X_train[text_col].astype(str)
    X_test[text_col] = X_test[text_col].astype(str)

    svc_random_grid = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
                       'kernel': ['rbf'], 'class_weight': ['balanced', None]}
    svc_param_grid = {'C': [10, 20, 30, 40], 'gamma': [0, 0.001, 0.008, 0.01, 0.1, 0.5]}
    ###############################################################################
    sgd_random_grid = {"loss": ["log"],
                       "alpha": [0.0001, 0.00001],
                       "penalty": ["elasticnet"],
                       "l1_ratio": 0.2 * np.arange(0, 5),
                       "shuffle": [True],
                       "learning_rate": ['optimal']}
    sgd_param_grid = []
    ###############################################################################
    lr_random_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    lr_param_grid = []

    # %%

    class_weights = 'balanced'
    classifiers = {
        "nb": {"name": "multinomial naive bayes", "clf": MultinomialNB()},
        "gnb": {"name": "gaussian naive bayes", "clf": GaussianNB()},
        "lr": {"name": "logistic regression", "clf": LogisticRegression(n_jobs=-1, class_weight=class_weights),
               "param_grid": lr_param_grid, "random_grid": lr_random_grid},
        "sgd": {"name": "stochastic gradient descent ",
                "clf": SGDClassifier(n_jobs=-1, class_weight=class_weights, loss='log'), "param_grid": sgd_param_grid,
                "random_grid": sgd_random_grid},
        "svm": {"name": "support vector machine ", "clf": SVC(probability=True, class_weight=class_weights),
                "param_grid": svc_param_grid, "random_grid": svc_random_grid}
    }
    methods = {"ovr": OneVsRestClassifier, "cc": ClassifierChain}

    # %%

    # functions
    def label_sentences(corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the post.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(doc2vec.TaggedDocument(str(v).split(), [label]))
        return labeled

    def get_vectors(model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.docvecs[prefix]
        return vectors

    def get_vectors_w2v(model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained word2vec model
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.wordvecs[prefix]
        return vectors

    def word_averaging(wv, words):
        all_words, mean = set(), []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in wv.vocab:
                mean.append(wv.syn0norm[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)

        if not mean:
            logging.warning("cannot compute similarity with no input %s", words)
            return np.zeros(wv.vector_size, )

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def word_averaging_list(wv, text_list):
        return np.vstack([word_averaging(wv, t) for t in text_list])

    def w2v_tokenize_text(text):
        tokens = []
        for sent in nltk.sent_tokenize(text, language='english'):
            for word in nltk.word_tokenize(sent, language='english'):
                if len(word) < 2:
                    continue
                tokens.append(word)
        return tokens

    # %%

    def write_in_file(algorithmName, result):
        f.write(algorithmName + ":\n")
        f.write(result)
        f.write('-----------------------------------------------------------------')
        f.write('\n')

    # %%

    # Classifiers
    def classify_multilabel(algorithm, featureMode, tuningMode):
        algorithmName = classifiers[algorithm]["name"]

        def update_dict(d):
            return {f'estimator__{k}': v for k, v in d.items()}

        #     model = "this will be our model!"
        if (tuningMode == 0):
            print('Tuning', tuningMode, '-running default settings...')
            model = methods[method](classifiers[algorithm]["clf"])
        elif (tuningMode == 1):
            print('Tuning', tuningMode, '-running grid search...')
            model = GridSearchCV(estimator=methods[method](classifiers[algorithm]["clf"]),
                                 param_grid=update_dict(classifiers[algorithm]["param_grid"]),
                                 cv=3, n_jobs=-1, verbose=2)
        elif (tuningMode == 2):
            print('Tuning', tuningMode, '-running default randomized search...')
            model = RandomizedSearchCV(estimator=methods[method](classifiers[algorithm]["clf"]),
                                       param_distributions=update_dict(classifiers[algorithm]["random_grid"]),
                                       n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

        report = ""
        y_pred_probab = []
        if (featureMode == 0):
            print('Feature', featureMode, '-running TF-IDF')
            algorithmName += " + ngram range " + str(ngramRange)
            model.fit(tfidf_x_train, y_train)
            y_pred_probab = model.predict_proba(tfidf_x_test)
        elif (featureMode == 1):
            print('Feature', featureMode, '-running D2V')
            algorithmName += " + min count " + str(minCount) + " + features number " + str(d2v_max_feat)
            model.fit(d2v_x_train, y_train)
            y_pred_probab = model.predict_proba(d2v_x_test)

        if (tuningMode):
            report = "\nbestparameters:\n" + str(model.best_params_) + '\n'

        print(report)
        with open(f'{models_path}/{algorithm}--{featureMode}--{tuningMode}--Multilabel.pkl', 'wb') as f:
            pickle.dump(model, f)
        return (algorithmName, model, y_pred_probab)

    if tokenizer == "tfidf":
        tfidf_vectorizer_title = TfidfVectorizer(
            stop_words='english',
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{2,}',
            ngram_range=ngramRange,
            max_features=tfidf_max_feat)

        tfidf_x_train = tfidf_vectorizer_title.fit_transform(X_train[text_col].values.astype('U'))
        tfidf_x_test = tfidf_vectorizer_title.transform(X_test[text_col].values.astype('U'))

    else:
        X_train_title = label_sentences(X_train[text_col], 'Train')
        X_test_title = label_sentences(X_test[text_col], 'Test')

        all_data_title = X_train_title + X_test_title

        model_dbow = Doc2Vec(dm=0, vector_size=d2v_max_feat, negative=5, min_count=minCount,
                             alpha=0.065, min_alpha=0.065, workers=40)
        model_dbow.build_vocab([x for x in tqdm(all_data_title)])

        for epoch in range(30):
            print(epoch)
            model_dbow.train(utils.shuffle([x for x in tqdm(all_data_title)]),
                             total_examples=len(all_data_title), epochs=1)
            model_dbow.alpha -= 0.002
            model_dbow.min_alpha = model_dbow.alpha

        d2v_x_train = get_vectors(model_dbow, len(X_train_title), d2v_max_feat, 'Train')
        d2v_x_test = get_vectors(model_dbow, len(X_test_title), d2v_max_feat, 'Test')

    def eval(y_original, y_pred, y_pred_probab):
        org_label_count_vec = np.sum(y_original, axis=1)
        repo_5_tags = len(np.where(org_label_count_vec >= 5)[0])

        def calc(p1, p2, p3, func, **kwargs):
            return func(p1, p2, **kwargs)

        def calc_prob(p1, p2, p3, func, **kwargs):
            return func(p1, p3, **kwargs)

        def success_rate(y_original, y_pred):
            common = 0
            for i in range(0, y_pred.shape[0]):
                if (sum(y_original.values[i] * y_pred[i])) > 0:
                    common = common + 1
            success = common / y_pred.shape[0]
            return success

        def coverage(y_original, y_pred):
            x = y_pred.sum(axis=0)
            c = np.count_nonzero(x > 0)
            cov = c / y_original.shape[1]
            return cov

        def prf_at_k(y_original, y_pred_probab):
            k_list = [1, 2, 3, 5, 8, 10]
            s1, s5 = {}, {}
            r, p, f = {}, {}, {}
            y_org_array = y_original.values

            for k in k_list:
                org_label_count = np.sum(y_org_array, axis=1).tolist()
                top_ind = []
                top_ind = np.argpartition(y_pred_probab, -1 * k, axis=1)[:, -1 * k:]
                pred_in_org = y_org_array[np.arange(y_org_array.shape[0])[:, None], top_ind]
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
            return r, p, f, s1, s2, s3, s4, s5

        r, p, f, s1, s5 = {}, {}, {}, {}, {}
        r, p, f, s1, s2, s3, s4, s5 = prf_at_k(y_original, y_pred_probab)

        metrics = {
            "Success_Rate": partial(calc, func=success_rate),
            "Coverage": partial(calc, func=coverage),
            "LRL": partial(calc, func=sklearn.metrics.label_ranking_loss),
            "F1_micro": partial(calc, func=sklearn.metrics.f1_score, average='micro'),
            "F1_macro": partial(calc, func=sklearn.metrics.f1_score, average='macro'),
            "F1_weighted": partial(calc, func=sklearn.metrics.f1_score, average='weighted'),
            "F1_samples": partial(calc, func=sklearn.metrics.f1_score, average='samples'),
            "P_micro": partial(calc, func=sklearn.metrics.precision_score, average='micro'),
            "P_macro": partial(calc, func=sklearn.metrics.precision_score, average='macro'),
            "P_weighted": partial(calc, func=sklearn.metrics.precision_score, average='weighted'),
            "P_samples": partial(calc, func=sklearn.metrics.precision_score, average='samples'),
            "R_micro": partial(calc, func=sklearn.metrics.recall_score, average='micro'),
            "R_macro": partial(calc, func=sklearn.metrics.recall_score, average='macro'),
            "R_weighted": partial(calc, func=sklearn.metrics.recall_score, average='weighted'),
            "R_samples": partial(calc, func=sklearn.metrics.recall_score, average='samples'),
            "Hamming_loss": partial(calc, func=sklearn.metrics.hamming_loss),
            "Exact_match_ratio": partial(calc, func=sklearn.metrics.accuracy_score),
            "AUC_micro": partial(calc, func=sklearn.metrics.roc_auc_score, average='micro'),
            "AUC_macro": partial(calc, func=sklearn.metrics.roc_auc_score, average='macro'),
            "AUC_wighted": partial(calc, func=sklearn.metrics.roc_auc_score, average='weighted'),
            "Coverage_err": partial(calc, func=sklearn.metrics.coverage_error),
            "Avg_P_score_micro": partial(calc, func=sklearn.metrics.average_precision_score, average='micro'),
            "Avg_P_score_macro": partial(calc, func=sklearn.metrics.average_precision_score, average='macro')
        }

        eval_results = {i: "{:.2f}".format(metrics[i](y_original, y_pred, y_pred_probab) * 100) for i in metrics}
        class_report = classification_report(y_original, y_pred)
        return eval_results, r, p, f, s1, s5, class_report

    feat = 0 if tokenizer == 'tfidf' else 1
    # change tuningmode here: default, randomized, gridsearch
    tune = 0
    # change model name here: nb, gnb, sgd, lr
    algo = model
    algorithmName, model, y_pred_probab = classify_multilabel(algo, feat, tune)

    # %%

    thr = [0.25, 0.5]
    for t in thr:
        print('\n------Threshold------', t)
        preds = np.where(y_pred_probab > t, 1, 0)
        results, r, p, f, s1, s5, class_report = eval(y_test, preds, y_pred_probab)
        print(results, r, p, f, s1, s5, class_report, sep="\n\n")
        print('\n---------------------')
