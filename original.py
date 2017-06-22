import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

plt.show()


# load the datasets
names = ['dealerId', 'gender']
dataset = pd.read_csv("C:/Users/eyein/Desktop/GenderPythonTest.csv", names=names)

df = DataFrame(dataset)

#replace the empty strings in gender with nan
df['gender'].replace('', np.nan, inplace=True)

#drop the null values in gender
df.dropna(subset=['gender'], inplace=True)

#convert gender to lower case
gender = df['gender'].map(lambda x: x if type(x)!=str else x.lower())


#insert a new column representing gender by 0=male, 1=female
df['gen_num'] = df['gender'].map(lambda x: 1 if x =='female' else 0)

#speficy predictor and target
y = df.gen_num.values

#fit the model
clf = svm.SVC(gamma=0.001, C=100.)
#clf.fit(X, y)


#dv = DictVectorizer()
#lr = LogisticRegression()
#vectorizer = CountVectorizer(min_df=1)

#tokenizing text in dealerId
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df.dealerId)

#Check dictionary indices
print(X_train_counts.shape)

#take occurrences to frequencies
#fit estimator to the data
tfidf_transformer = TfidfTransformer()

#tranform count-matrix to a tf-idf representation
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

#training Naice Bayers classifier
clf = MultinomialNB().fit(X_train_tfidf, y)

#input new dealerId, extract features
docs_new = ['teQCUniversity', 'alexisnihon', 'teQCSteFoy', 'mamaxiforme1', 'teONStockyard', 'carrefourcharlesbourg']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#predict outcome
predicted = clf.predict(X_new_tfidf)

#print result
for doc, row in zip(docs_new, predicted):
    print('%r => %s' % (doc, y[row]))

#evaluating the prediction
 