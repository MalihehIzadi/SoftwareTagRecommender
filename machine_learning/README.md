# The scripts and files for our machine-learning based models

## Requirements

To install all the requirements:

`$ pip install -r requirements`


## Traditional Classification
Here are the scripts for multi-label multi-class classification using different techniques.
We have traditional classifiers including:
1. MNB: Multinomial Naive Bayes
2. GNB: Gaussian Naive Bayes
3. SGD: Stochastic Gradient Descent
4. LR: Logistic Regression
5. FastText

### Traditional-ML Approach
For training and evaluating with MNB,GNB,SGD or LR algorithms use `traditional_classifiers.py`

```
$ python traditional_classifiers.py --help

Usage: traditional_classifiers.py [OPTIONS]

  Train and Evaluation on Traditional ML Models.

Options:
  --train TEXT                 train CSV file
                               path.

  --test TEXT                  test CSV file path.
  --topics_column TEXT         The name of topics
                               column.

  --readme_column TEXT         The name of readme
                               text column.

  --model [nb|gnb|lr|sgd|svm]  Model Type.
  --tokenizer [tfidf|doc2vec]  tokenizer Type.
  --method [ovr|cc]            ClassifierChain or OneVsRest
  --help                       Show this message
                               and exit.
```

### FastText
For training and evaluating with [fastText](https://github.com/facebookresearch/fastText) model use `fasttext_classifier.py`

```
$ python fasttext_classifier --help

Usage: fasttext_classifier.py [OPTIONS]

  Train and Evaluate data on fastText model

Options:
  --train TEXT           train CSV file path.
  --test TEXT            test CSV file path.
  --topics_column TEXT   The name of topics
                         column.

  --readme_column TEXT   The name of readme text
                         column.

  --model_output TEXT    Model save path.
  --learning_rate FLOAT  Learning rate Value.
  --epoch INTEGER        Number of Epoch.
  --word_ngrams INTEGER  Number of wordNgrams.
  --help                 Show this message and
                         exit.
```


## Transformers-based Classification

We also have transformers-based models. We use the following pre-trained models and fine-tune them on our data for multi-label classification.
1. BERT base by Google AI Language,
2. RoBERTa by Facebook AI,
3. ALBERT by Google Research and Toyota,
4. DistilBERT by HuggingFace,
5. XLM by Facebook AI,
6. XLNet by CMU and Google AI Brain,

## Multi-hot Encoded Conversion
Furthermore, `multihot_encoded_conversion.py` converts the list of labels for each data point (repository) to a multi-hot encoded list with the length of labels (tags) in the dataset to prepare for classification.

```
$ python multihot_encoded_conversion.py --help

Usage: multihot_encoded_conversion.py [OPTIONS]

  Convert a csv file with comma separated topics
  to multi-hot encoded

Options:
  -i TEXT  Input CSV file path.
  -t TEXT  The topics column name.
  -o TEXT  Output CSV file path.
  --help   Show this message and exit.```