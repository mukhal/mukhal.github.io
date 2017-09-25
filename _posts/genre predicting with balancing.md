
## Predicting movie genre from its title


```python
import numpy, pandas as pd
import sklearn
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
data = pd.read_csv('movies.csv',quotechar='"')
```


```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



## Preprocessing data


```python
# drop non ascii titles
def is_ascii(s):
    return all(ord(c) < 128 for c in s)

data = data.drop(data[data['title'].apply(lambda t: not is_ascii(t))].index)

```

### Processing title 
We strip away numbers, parenthesis... etc


```python
import re
def process_title(title): 
    # strip away numbers and parenthesis
    title = title.replace('(','').replace(')','')
    title = re.sub(r'\d+','',title)
    # strip away "part" word
    title = re.sub(r'[Pp]art','',title)
    #strip II and III and IV
    title = title.replace('II','').replace('III','').replace('IV','')
    return title

data['title'] = data['title'].apply(process_title) 
#drop empty titles
data = data.drop(data[data['title'].str.strip() ==''].index)
```

### Converting to binary classification
This is a multilabel classification problem, we will convert it to set of binary classification problems 


```python
# drop movies with no genres
data['genres'] = data['genres'].apply(lambda gs:gs.lower())

# get all genres
genres = set()
for gs in data['genres'].str.split('|'):
    genres |= set(gs)
genres.remove('(no genres listed)')

for g in genres:
    data[g] = data['genres'].apply(lambda gs: 1 if g in gs.split('|') else 0)
```


```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>sci-fi</th>
      <th>horror</th>
      <th>fantasy</th>
      <th>adventure</th>
      <th>western</th>
      <th>musical</th>
      <th>children</th>
      <th>...</th>
      <th>romance</th>
      <th>film-noir</th>
      <th>crime</th>
      <th>drama</th>
      <th>animation</th>
      <th>action</th>
      <th>comedy</th>
      <th>documentary</th>
      <th>war</th>
      <th>imax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>adventure|animation|children|comedy|fantasy</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>adventure|children|fantasy</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>comedy|romance</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>comedy|drama|romance</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride</td>
      <td>comedy</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



#### Converting to lower case 


```python
data['title']=data['title'].apply(lambda t: t.lower())
```

### Checking class distribution


```python
d = dict(data.mean())
del d['movieId']
```


```python
#sorting genres by frequency occurence
g_sorted_freq = sorted(d.keys(),key=lambda x:d[x])

# dropping the 6 least common genres
for g in g_sorted_freq[:6]:
    data = data.drop(g,axis=1)
    genres.remove(g)
```


```python
# checking classes distribution
plt.ylim((0,1.0))
plt.ylabel('portion of positive examples')
data[list(genres)].mean().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x4018f908>




![png](genre%20predicting%20with%20balancing_files/genre%20predicting%20with%20balancing_16_1.png)


We can see we have very imbalanced data with less than 10% positive examples in about 6 classes.
We will have to deal with this to be able to evaluate or models correctly

#### creating balanced dataset for each genre by means of undersampling


```python
balanced_data = {}
for g in genres:
    positive_examples = data[data[g]==1]
    negative_examples = data[data[g]==0].sample(n=len(positive_examples.index))
    balanced_data[g] = positive_examples.append(negative_examples)
    
```

## Treating it as text classification using Naive Bayes


```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, train_size = 0.6)
```


```python
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def learn_counts(train, target):
    cnt_word_given_class = defaultdict(lambda: defaultdict(lambda :0))
    for i,row in train.iterrows():
        classes = train[target].unique()
        for word in tokenizer.tokenize(row['title']):
            cnt_word_given_class[word][row[target]]+=1.0
    cnt_classes = {c:len(train[train[target]==c].index) for c in classes}
    V = len(cnt_word_given_class.keys())
    return classes, cnt_classes, cnt_word_given_class, V 

def get_class_prob_given_word(word,cnt_w_c, cnt_c,classes,K):
    return {c: (1.0*K + cnt_w_c[word][c])  / (cnt_c[c] + K*V) for c in classes} 

def get_text_class (text,cnt_w_c, cnt_c, V, classes=[0,1],K=1):
    probs = {c:0 for c in classes}
    for word in tokenizer.tokenize(text):
            word_probs = get_class_prob_given_word(word,cnt_w_c,cnt_c,classes,K)
            for c in probs:
                probs[c] += math.log(word_probs[c])
    
    return max(probs.keys(), key=lambda x:probs[x])
        
```


```python
f1_scores = []
for g in genres:
    train,test = train_test_split(balanced_data[g],train_size = 0.6)
    classes,cnt_c,cnt_w_c, V = learn_counts(train,g)
    y_pred = test['title'].apply(lambda t: get_text_class(t,cnt_w_c,cnt_c, V))
    f1_scores.append(f1_score(y_pred,test[g]))
    print 'for genre %s , f1 score is %.2f' %(g, f1_scores[-1])
    
print 'average f1 score over all genres : %.2f' %(np.mean(f1_scores))
```

    for genre sci-fi , f1 score is 0.71
    for genre horror , f1 score is 0.70
    for genre fantasy , f1 score is 0.66
    for genre adventure , f1 score is 0.67
    for genre thriller , f1 score is 0.56
    for genre mystery , f1 score is 0.64
    for genre romance , f1 score is 0.61
    for genre crime , f1 score is 0.62
    for genre drama , f1 score is 0.50
    for genre action , f1 score is 0.64
    for genre comedy , f1 score is 0.58
    for genre documentary , f1 score is 0.67
    for genre war , f1 score is 0.65
    average f1 score : 0.63
    

### Classification using word embeddings


```python
# glove word embeddings
import numpy as np

embeddings = {}
with open('glove.6B/glove.6B.50d.txt', 'r') as f:
    for line in f:
        embeddings[line.split()[0]] = np.array(map(float, line.split()[1:]))
```


```python
# transform text (a title) to an embedding by averaging word embeddings


def get_mean_embeddings(docs,embeddings):
    means = []
    dim = len(embeddings.values()[0])
    for doc in docs :
        words = tokenizer.tokenize(doc)
        means.append(np.mean([embeddings[w] if w in embeddings else np.zeros(dim) for w in words], axis=0)) 
    return np.array(means)
```


```python
def get_mean_embeddings(docs,embeddings):
    dim = len(embeddings.values()[0])
    return np.array([
                np.mean([embeddings[w]
                         for w in tokenizer.tokenize(doc) if w in embeddings] or
                        [np.zeros(dim)], axis=0)
                for doc in docs
            ])
```

### Trying out different models  (SVM , Logistic Regression, KNN, Random Forests)


```python
import sklearn.svm as svm
from sklearn.metrics import f1_score
clf = svm.SVC(kernel='rbf')
f1_scores = []
for g in genres:
    genre_data = balanced_data[g]
    train,test = train_test_split(genre_data,train_size = 0.6)
    train_feature_matrix = get_mean_embeddings(train['title'],embeddings)
    test_feature_matrix = get_mean_embeddings(test['title'],embeddings)
    clf.fit(train_feature_matrix,train[g])
    y_pred = clf.predict(test_feature_matrix)
    f1_scores.append(f1_score(test[g],y_pred))
    print 'for "%s" , f1 score = %.2f' %(g,f1_scores[-1])
    
print 'average f1 score over all genres : %.2f ' %(np.mean(f1_scores))
```

    for "sci-fi" , f1 score = 0.70
    for "horror" , f1 score = 0.68
    for "fantasy" , f1 score = 0.62
    for "adventure" , f1 score = 0.66
    for "thriller" , f1 score = 0.63
    for "mystery" , f1 score = 0.58
    for "romance" , f1 score = 0.62
    for "crime" , f1 score = 0.56
    for "drama" , f1 score = 0.59
    for "action" , f1 score = 0.67
    for "comedy" , f1 score = 0.62
    for "documentary" , f1 score = 0.64
    for "war" , f1 score = 0.65
    average f1 score over all genres : 0.63 
    


```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
for g in genres:
    genre_data = balanced_data[g]
    train,test = train_test_split(genre_data,train_size = 0.6)
    train_feature_matrix = get_mean_embeddings(train['title'],embeddings)
    test_feature_matrix = get_mean_embeddings(test['title'],embeddings)
    clf.fit(train_feature_matrix,train[g])
    y_pred = clf.predict(test_feature_matrix)
    f1_scores.append(f1_score(test[g],y_pred))
    print 'for "%s" , f1 score = %.2f' %(g,f1_scores[-1])
    
print 'average f1 score over all genres : %.2f ' %(np.mean(f1_scores))
```

    for "sci-fi" , f1 score = 0.66
    for "horror" , f1 score = 0.68
    for "fantasy" , f1 score = 0.62
    for "adventure" , f1 score = 0.65
    for "thriller" , f1 score = 0.62
    for "mystery" , f1 score = 0.57
    for "romance" , f1 score = 0.60
    for "crime" , f1 score = 0.59
    for "drama" , f1 score = 0.57
    for "action" , f1 score = 0.66
    for "comedy" , f1 score = 0.61
    for "documentary" , f1 score = 0.62
    for "war" , f1 score = 0.66
    average f1 score over all genres : 0.60 
    


```python
import sklearn.neighbors
clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)
for g in genres:
    genre_data = balanced_data[g]
    train,test = train_test_split(genre_data,train_size = 0.6)
    train_feature_matrix = get_mean_embeddings(train['title'],embeddings)
    test_feature_matrix = get_mean_embeddings(test['title'],embeddings)
    clf.fit(train_feature_matrix,train[g])
    y_pred = clf.predict(test_feature_matrix)
    f1_scores.append(f1_score(test[g],y_pred))
    print 'for "%s" , f1 score = %.2f' %(g,f1_scores[-1])
    
print 'average f1 score over all genres : %.2f ' %(np.mean(f1_scores))
```

    for "sci-fi" , f1 score = 0.65
    for "horror" , f1 score = 0.68
    for "fantasy" , f1 score = 0.63
    for "adventure" , f1 score = 0.64
    for "thriller" , f1 score = 0.52
    for "mystery" , f1 score = 0.55
    for "romance" , f1 score = 0.54
    for "crime" , f1 score = 0.48
    for "drama" , f1 score = 0.48
    for "action" , f1 score = 0.59
    for "comedy" , f1 score = 0.56
    for "documentary" , f1 score = 0.63
    for "war" , f1 score = 0.59
    average f1 score over all genres : 0.61 
    


```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20)
for g in genres:
    genre_data = balanced_data[g]
    train,test = train_test_split(genre_data,train_size = 0.6)
    train_feature_matrix = get_mean_embeddings(train['title'],embeddings)
    test_feature_matrix = get_mean_embeddings(test['title'],embeddings)
    clf.fit(train_feature_matrix,train[g])
    y_pred = clf.predict(test_feature_matrix)
    f1_scores.append(f1_score(test[g],y_pred))
    print 'for "%s" , f1 score = %.2f' %(g,f1_scores[-1])
    
print 'average f1 score over all genres : %.2f ' %(np.mean(f1_scores))
```

    for "sci-fi" , f1 score = 0.63
    for "horror" , f1 score = 0.64
    for "fantasy" , f1 score = 0.61
    for "adventure" , f1 score = 0.60
    for "thriller" , f1 score = 0.58
    for "mystery" , f1 score = 0.53
    for "romance" , f1 score = 0.54
    for "crime" , f1 score = 0.57
    for "drama" , f1 score = 0.54
    for "action" , f1 score = 0.64
    for "comedy" , f1 score = 0.58
    for "documentary" , f1 score = 0.59
    for "war" , f1 score = 0.65
    average f1 score over all genres : 0.59 
    

### Conclusion

Ok. This was an attempt to predict movies genres using titles only. At first we explored Naive Bayes. Then we introduced word embeddings where we used glove word embeddings to obtain an embedding for the whole title by averaging word embeddings of the constituent words title word embeddings. We used an SVM with RBF as kernel, Logistic regression model, KNN, and Random Forests.

We can see that best models on average for all genres are Naive Bayes, SVM and Logistic Regression.
However these model may vary on individual genres.

We also saw how to deal with imbalanced data by performing undersampling. 



```python

```
