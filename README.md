# Twitter Sentiment Analysis
There are many tweets gets uploaded out there on twitter, few of them might be sensitive or let say those tweets has negative sentiments. 
So our objective is to create a model that can detect such tweets and is able to label them as positive or negatie tweets with better accuracy.
To train our model we have data set that contains 1.6 million tweets in it, **tweets with positive sentiments are labeled as '4' and tweet with negative sentiments labeled with '4'.**
## Requirements
python >=2.7  

*Install libraries :*
```
pip install sklearn 
pip install numpy 
pip install pandas
pip install scipy
pip install seaborn
pip install matplotlib
pip install WordCloud
pip install nltk
pip install scikit-learn
```

## Detail about the project
SO, let's start and see how we are going to proceed in order to achieve our goal and that is, training our model so that it can detect the tweets with positive and negative sentiments.

Whenever we work on large dataset, the first thing we do is, exploring the data, and try to get some information out of it, in our case we will try to visualize the data. 
This is the order in which I worked on this project.
1. load the data
2. remove unwanted data,columns etc. for example, columns the we aren't going to use or we see or has little or no importance , we will delete it or remove it from our dataframe. We just don't want any unwanted unnecessary information in our dataframe.
3. Check whether the data has null values or not, if it has, then we will try to fix that, we might drop the column that has missing values(but we can't do that in every case), so we can use imputation method. Luckily in my case, the data has no missing values.
4. Here comes the visualization part, we will try to visualize the data with the help of word cloud,there are other methods too. I will upload a separate code using different techniues and algorithms in order to get more accuracy.
5. filtering the data, in our case we will fiter the tweets for example. 
these are the things that we are going to remove from our tweets.
```
removing punctuation
removing numbers 
removing hyperlinks 
removing stop words 
stemming  
lemmatization
```
**We will use NLTK library for all this.**

6. Feature extraction
I have used count vectorizer for feature extraction, there are ways of doing the same such as tfidfvectorizer, word embedding etc.
7. Spilliting our data set into test and training data. 
8. Now I have used naive bayes model to train and test our data, you might try to do the same using some other algorithm.
9. Scoring and Matrix
  I have used confusion matrix to visualize my model's result.
