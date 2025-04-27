# Predicting Real Disaster Tweets with a Wide & Deep Network

`tweet.ipynb` contains a solution to the Kaggle Competition: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview) - Predict which Tweets are about real disasters and which ones are not.

The best trained model is saved as `best_model.pth`, and the test set predictions are available in `submission.csv`.

## Competition Description (from Kaggle)

Twitter has become an important communication channel in times of emergency.

The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

## Dataset Description (from Kaggle)

All data can be downloaded from [Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data)

Each sample in the train and test set has the following information:

- The `text` of a tweet
- A `keyword` from that tweet (although this may be blank!)
- The `location` the tweet was sent from (may also be blank)

You are predicting whether a given tweet is about a real disaster or not. If so, predict a `1`. If not, predict a `0`.

- In train.csv only, `target` denotes whether a tweet is about a real disaster (1) or not (0)

## Model Overview

This solution employs a Wide & Deep architecture:

- Wide Network: A single-layer linear model (logistic regression) leveraging TF-IDF features (1-gram and 2-gram).
- Deep Network: A two-layer Transformer encoding tokenized tweet content.
- The final prediction is a weighted combination of outputs from both networks, aiming to integrate the memorization ability of the wide model with the generalization power of the deep model.

## Preprocessing Steps

- Concatenate `text`, `keyword`, and `location`.
- Convert to lowercase, remove non-alphabetic characters.
- Remove stopwords and perform lemmatization.
- Calculate TF-IDF features for each tweet.
- Tokenize text: Index `0` reserved for `<PAD>`, `1` for `<UNK>`.

## Evaluation

Model performance is evaluated using the [F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) between the predicted and expected answers, as per the competition guidelines.
  
