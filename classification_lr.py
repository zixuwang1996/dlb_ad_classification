from sklearn  import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text  import TfidfVectorizer, CountVectorizer
from sklearn  import decomposition, ensemble
from sklearn.feature_extraction  import text
from sklearn.metrics  import precision_recall_fscore_support
import pandas, numpy, string

# create a dataframe using texts and lables
trainDF = pandas.read_csv("data/crate-new/test.csv", delimiter='\t', error_bad_lines=False)
my_stop_words = text.ENGLISH_STOP_WORDS

# split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


# Create different features
# 1. create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=my_stop_words)
count_vect.fit(trainDF['text'])
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# 2. create a TF-IDF vectorizer object
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000, stop_words=my_stop_words)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3),max_features=5000, stop_words=my_stop_words)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3),max_features=5000, stop_words=my_stop_words)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)


def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    return precision_recall_fscore_support(valid_y, predictions, average='binary'), classifier

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n +  1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print(coef_1, fn_1, coef_2, fn_2)

# Naive Bayes on Count Vectors
accuracy, clf = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print('NB, Count Vectors:')
print(accuracy)
show_most_informative_features(count_vect, clf)

# Naive Bayes on Word Level TF IDF Vectors
accuracy, clf = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel TF-IDF: ")
print(accuracy)
show_most_informative_features(tfidf_vect, clf)


# Linear Classifier on Count Vectors
accuracy, clf = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ")
print(accuracy)
show_most_informative_features(count_vect, clf)

# Linear Classifier on Word Level TF IDF Vectors
accuracy, clf = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ")
print(accuracy)
show_most_informative_features(tfidf_vect, clf)


# SVM on Count Vectors
accuracy, _ = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
print("SVM, Count Vectors: ")
print(accuracy)

# SVM on Ngram Level TF IDF Vectors
accuracy, _ = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
print("SVM, WordLevel Vectors: ")
print(accuracy)

# RF on Count Vectors
accuracy, _ = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print("RF, Count Vectors: ")
print(accuracy)

# RF on Word Level TF IDF Vectors
accuracy, _ = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("RF, WordLevel TF-IDF: ")
print(accuracy)
