import json
from datetime import datetime
# read the data from disk and split into lines
# we use .strip() to remove the final (empty) line
with open("yelp_academic_dataset_review.json") as f:
    reviews = f.read().strip().split("\n")

# each line of the file is a separate JSON object
reviews = [json.loads(review) for review in reviews[:200000]]

# we're interested in the text of each review
# and the stars rating, so we load these into
# separate lists
texts = [review['text'] for review in reviews]
stars = [review['stars'] for review in reviews]


from collections import Counter

def balance_classes(xs, ys):
    """Undersample xs, ys to balance classes."""
    freqs = Counter(ys)

    # the least common class is the maximum number we want for all classes
    max_allowable = freqs.most_common()[-1][1]
    num_added = {clss: 0 for clss in freqs.keys()}
    new_ys = []
    new_xs = []
    for i, y in enumerate(ys):
        if num_added[y] < max_allowable:
            new_ys.append(y)
            new_xs.append(xs[i])
            num_added[y] += 1
    return new_xs, new_ys


print(Counter(stars))
balanced_x, balanced_y = balance_classes(texts, stars)
print(Counter(balanced_y))

from sklearn.feature_extraction.text import TfidfVectorizer

# This vectorizer breaks text into single words and bi-grams
# and then calculates the TF-IDF representation
vectorizer = TfidfVectorizer(ngram_range=(1,2))
print "finished TFid"
t1 = datetime.now()

# the 'fit' builds up the vocabulary from all the reviews
# while the 'transform' step turns each indivdual text into
# a matrix of numbers.
vectors = vectorizer.fit_transform(balanced_x)
print "finished vectorizer:"
print(datetime.now() - t1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectors, balanced_y, test_size=0.33, random_state=42)

from sklearn.svm import LinearSVC
from sklearn import linear_model

# initialise the SVM classifier
#classifier = LinearSVC()
classifier = linear_model.LinearRegression()

# train the classifier
print "start training:"
t1 = datetime.now()
print(datetime.now() - t1)
classifier.fit(X_train, y_train)
print "end training:"
print(datetime.now() - t1)

preds = classifier.predict(X_test)
preds[preds > 5.0] = 5.0
preds[preds < 1.0] = 1.0
print(list(preds[:10]))
print(y_test[:10])

from sklearn import metrics
print "mean squared error"
print(metrics.mean_squared_error(y_test, preds))
print "r2 score"
print(metrics.r2_score(y_test, preds))
print "explained_variance_score"
print(metrics.explained_variance_score(y_test, preds))
print "mean_absolute_error"
print(metrics.mean_absolute_error(y_test, preds))


print(texts[0])
print(texts[4])

#Classifier

# keep = set([1, 2, 4, 5])
#
# # calculate the indices for the examples we want to keep
# keep_train_is = [i for i, y in enumerate(y_train) if y in keep]
# keep_test_is = [i for i, y in enumerate(y_test) if y in keep]
#
# # convert the train set
# X_train2 = X_train[keep_train_is, :]
# y_train2 = [y_train[i] for i in keep_train_is]
# y_train2 = ["n" if (y == 1 or y == 2) else "p" for y in y_train2]
#
# # convert the test set
# X_test2 = X_test[keep_test_is, :]
# y_test2 = [y_test[i] for i in keep_test_is]
# y_test2 = ["n" if (y == 1 or y == 2) else "p" for y in y_test2]
#
#
# classifier.fit(X_train2, y_train2)
# preds = classifier.predict(X_test2)
#
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, preds))
#
# from sklearn.metrics import classification_report
# print(classification_report(y_test, preds))
#
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_test, preds))
#
# print(classification_report(y_test2, preds))
# print(confusion_matrix(y_test2, preds))
#
# # only two texts as an example
# texts = ["I really hated my stay at The NotARealName Hotel", "Had a really really great stay at NotARealName - would recommend to everyone"]
#
# # note that we only call .transform() here and not .fit_transform()
# # as we want to keep the vocabulary from the previous experiments
# vecs = vectorizer.transform(texts)
#
# # predict a positive or negative label for each input
# print(classifier.predict(vecs))
