import pandas as pd
import numpy as np
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import nltk  # Natural Language tool kit
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
#from jupyterthemes import jtplot

##import the data
tweet_data = pd.read_csv(r'training.1600000.processed.noemoticon.CSV',engine = 'python')

#column heading
tweet_data.columns = ["label", "id", "date_time", "query","username","tweets"]

#dropping unnecessary columns
tweet_data = tweet_data.drop(['id','date_time','query'],axis=1)

#create a list of all tweets
all_tweets = tweet_data['tweets'].tolist()
#len(all_tweets)

#positive tweets
positive_tweets = tweet_data[tweet_data['label']==4]
print(positive_tweets)

#negative tweets
negative_tweets = tweet_data[tweet_data['label']==0]
print(negative_tweets)


#creating a string containing all the tweets
#string = ''
#for ele in all_tweets[:]:
#    string+=ele
#print(string)


massive_string = ' '.join(all_tweets)   #or we can use a very simple code   ###

#just to have a visual representation of the texts in the dataset

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(massive_string))   #visulaize the data

#let's visualize negative tweets or let say tweet with negative sentiments
negative_tweets_list = negative_tweets['tweets'].tolist()
negative_string = ' '.join(negative_tweets_list)
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(negative_string))

#let's do the same with positive tweets
positive_tweets_list = positive_tweets['tweets'].tolist()
positive_string = ' '.join(positive_tweets_list)
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(positive_string))



# DATA CLEANING - Removing STOPWORDS and punctuation

print(string.punctuation)



nltk.download("stopwords")
stopwords.words('english')


# Test = 'Good morning beautiful people :)... I am having fun learning Machine learning and AI!!'
# Test_punc_removed = [char for char in Test if char not in string.punctuation]
# Test_punc_removed

# Test_punc_removed_join = ''.join(Test_punc_removed)
# Test_punc_removed_join

# punctuation = string.punctuation
# punc_list = list(punctuation)
# Test_list = list(Test)
# Test_punc_remove = [char for char in Test_list if char not in punc_list]
# Test_punc_remove_string = ''.join(Test_punc_remove)
# Test_punc_remove_string

# Test_punc_removed_join_clean =[word for word in Test_punc_removed_join.split() if word.lower() not in  stopwords.words('english')]

def cleaning_data(every_tweet):
    # removing punctuation
    tweet_removed_punc = ''.join([char for char in every_tweet if char not in string.punctuation])
    # removing numbers
    tweet_removed_num = ''.join([char for char in tweet_removed_punc if char not in '1234567890'])
    # convert from uppercase to lowercase
    tweet_aftr_converted_to_Lowercase = ''.join([char.lower() for char in tweet_removed_num])
    # lemmatization
    lem_word_tokens = nltk.word_tokenize(tweet_aftr_converted_to_Lowercase)
    lemmatized_tweet = ''.join([wordnet_lemmatizer.lemmatize(word) for word in lem_word_tokens])
    # stemming
    stemming_word_tokens = nltk.word_tokenize(lemmatized_tweet)
    stemmed_tweet = ''.join([snowball_stemmer.stem(word) for word in stemming_word_tokens])
    # stop words
    stopwords_tokens = nltk.word_tokenize(stemmed_tweet)
    cleaned_tweets = ''.join([word for word in stopwords_tokens if word not in stopwords.words('english')])
    return cleaned_tweets

# tweet_data_cleaned = tweet_data['tweets'].apply(cleaning_data)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

######### feature extraction with tfidfvectorizer
tf_idfvectorizer = TfidfVectorizer(analyzer = cleaning_data)
tweets_tfidfvectorizer = tf_idfvectorizer.fit_transform(tweet_data['tweets'])
#print(tweets_tfidfvectorizer.get_feature_names())


########### feature extraction with countvectorizer
#countvectorizer = CountVectorizer(analyzer = cleaning_data)
#tweets_countvectorizer = countvectorizer.fit_transform(tweet_data['tweets'])
#print(tweets_countvectorizer.get_feature_names())
#X = tweets_countvectorizer


X = tweets_tfidfvectorizer
y = tweet_data['label']
#spilliting data in two part, first as
# training data and second as validation or test data


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##########Just to check whether the classifier works with sparse vector or not##########
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import csr_matrix
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target

for CLF in [GaussianNB, MultinomialNB, BernoulliNB, LogisticRegression, LinearSVC, GradientBoostingClassifier]:
    print(CLF.__name__, end='')
    try:
        CLF().fit(csr_matrix(X), y == 0)
        print(' PASS')
    except TypeError:
        print(' FAIL')



#using naive bayes classifier

#NB_classifier = BernoulliNB()
#NB_classifier.fit(X_train,y_train)

## using multinomial naive bayes classifier
NB_classifier2 = MultinomialNB()
NB_classifier2.fit(X_train,y_train)

#print("accuracy of test data prediction :",NB_classifier.score(X_train,y_train,sample_weight=None),'\n')
#print("accuracy of test data prediction :",NB_classifier.score(X_test,y_test,sample_weight=None))

print("accuracy of test data prediction2 :",NB_classifier2.score(X_train,y_train,sample_weight=None),'\n')  #0.850
print("accuracy of test data prediction2 :",NB_classifier2.score(X_test,y_test,sample_weight=None))         #0.766


##confusion metrix for visualizing the model's result
from sklearn.metrics import classification_report, confusion_matrix
# Predicting the Test set results
y_predict_test = NB_classifier2.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))


#### tf-idfvectorizer + MultinomialNB algorithm gives better results and accuracy
