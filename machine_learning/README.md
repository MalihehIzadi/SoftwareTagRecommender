# The scripts and files for our machine-learning based models
Here are the scripts for multi-label multi-class classification using different techniques.
We have traditional classifiers including:
1. MNB: Multinomial Naive Bayes,
2. GNB: Gaussian Naive Bayes
3. SGD: Stochastic Gradient Descent
4. LR: Logistic Regression
5. FastText

Using two different transformation methods; TF-IDF and Doc2Vec.

We also have transformers-based models. We use the following pre-trained models and fine-tune them on our data for multi-label classification.
1. BERT base by Google AI Language
2. RoBERTa by Facebook AI
3. ALBERT by Google research, Toyota
4. DistilBERT by HuggingFace
5. XLM by Facebook AI
6. XLNet by CMU, Google AI Brain

Furthermore, `multihot_encoded_conversion.ipnyb` script converts the list of labels for the data points (repositories) to a multi-hot encoded matrix with the length of labels (tags) in the dataset to prepare for classification.