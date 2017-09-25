---
layout: post
title: My first post!!!!
---
## Predicting movie genre from its title


```python
import numpy, pandas as pd
import sklearn
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



### drop non ascii titles


```python
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

### Converting to lower case 


```python
data['title']=data['title'].apply(lambda t: t.lower())
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
      <th>thriller</th>
      <th>mystery</th>
      <th>romance</th>
      <th>crime</th>
      <th>drama</th>
      <th>action</th>
      <th>comedy</th>
      <th>documentary</th>
      <th>war</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>toy story</td>
      <td>adventure|animation|children|comedy|fantasy</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>jumanji</td>
      <td>adventure|children|fantasy</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>grumpier old men</td>
      <td>comedy|romance</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>waiting to exhale</td>
      <td>comedy|drama|romance</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>father of the bride</td>
      <td>comedy</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Treating it as text classification using Naive Bayes


```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, train_size = 0.6)
```


```python
from collections import defaultdict
from nltk.tokenize import word_tokenize
genres = train['main_genre'].unique()

# genre probability


p_genres = {}
count_genres ={}
for g in genres:
    count_genres[g] = len(train[train['main_genre'] == g].index)
    p_genres[g] = count_genres[g]*1.0 / len(train.index)

prob_word_given_genre = defaultdict(lambda: defaultdict(lambda :0))

#compute counts onlu
for i,row in train.iterrows():
    words_in_title = word_tokenize(row['title'])
    for word in words_in_title:
        prob_word_given_genre[word][row['main_genre']]+=1.0


```


```python
import math
K = 0.05 # smoothing factor
V = len(prob_word_given_genre) # vocabulary size

def get_prob_genre_given_word(genre, word):
    return ( (K + prob_word_given_genre[word][genre]) * p_genres[genre]) / (count_genres[g] + K*V)
    

def get_genre_for_title(title):
    title = title.lower()
    probs = defaultdict(lambda :0.0)
    for w in word_tokenize(title):
        for g in genres:
            probs[g] += math.log(get_prob_genre_given_word(g,w))
    
    return max(probs.keys(),key=lambda k:probs[k])
```

### Testing accuracy



```python
from sklearn.metrics import accuracy_score, classification_report
y_pred = test['title'].apply(get_genre_for_title)
```


```python
accuracy_score(test['main_genre'],y_pred)
print classification_report(test['main_genre'],y_pred)
```

                 precision    recall  f1-score   support
    
         action       0.53      0.14      0.23      1711
      adventure       0.36      0.02      0.04       593
      animation       0.22      0.01      0.02       247
       children       0.00      0.00      0.00       208
         comedy       0.37      0.41      0.39      3196
          crime       0.14      0.01      0.01       781
    documentary       0.12      0.01      0.02      1144
          drama       0.32      0.76      0.45      3619
        fantasy       0.00      0.00      0.00        53
      film-noir       0.00      0.00      0.00        16
         horror       0.50      0.04      0.08       676
        musical       0.00      0.00      0.00        45
        mystery       0.00      0.00      0.00        71
        romance       0.00      0.00      0.00        68
         sci-fi       0.00      0.00      0.00        85
       thriller       0.00      0.00      0.00       132
            war       0.00      0.00      0.00        12
        western       0.00      0.00      0.00       105
    
    avg / total       0.32      0.34      0.26     12762
    
    

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
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

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


```python
train_feature_matrix = get_mean_embeddings(train['title'],embeddings)
test_feature_matrix = get_mean_embeddings(test['title'],embeddings)
```


```python

```


```python
len(train_feature_matrix)
```




    15451




```python
import sklearn.svm as svm

clf = svm.SVC(kernel='linear')
for g in genres:
    clf.fit(train_feature_matrix,train[g])
    y_pred = clf.predict(test_feature_matrix)
    print 'for genre ', g
    print classification_report(test[g],y_pred)
    
```

    for genre  sci-fi
                 precision    recall  f1-score   support
    
              0       0.93      1.00      0.97      9626
              1       0.00      0.00      0.00       675
    
    avg / total       0.87      0.93      0.90     10301
    
    for genre  horror
                 precision    recall  f1-score   support
    
              0       0.90      1.00      0.95      9306
              1       0.00      0.00      0.00       995
    
    avg / total       0.82      0.90      0.86     10301
    
    for genre  fantasy
                 precision    recall  f1-score   support
    
              0       0.95      1.00      0.97      9783
              1       0.00      0.00      0.00       518
    
    avg / total       0.90      0.95      0.93     10301
    
    for genre  adventure
                 precision    recall  f1-score   support
    
              0       0.91      1.00      0.95      9401
              1       0.00      0.00      0.00       900
    
    avg / total       0.83      0.91      0.87     10301
    
    for genre  thriller
                 precision    recall  f1-score   support
    
              0       0.84      1.00      0.92      8691
              1       0.00      0.00      0.00      1610
    
    avg / total       0.71      0.84      0.77     10301
    
    for genre  mystery
                 precision    recall  f1-score   support
    
              0       0.94      1.00      0.97      9728
              1       0.00      0.00      0.00       573
    
    avg / total       0.89      0.94      0.92     10301
    
    for genre  romance
                 precision    recall  f1-score   support
    
              0       0.85      1.00      0.92      8764
              1       0.00      0.00      0.00      1537
    
    avg / total       0.72      0.85      0.78     10301
    
    for genre  crime
                 precision    recall  f1-score   support
    
              0       0.89      1.00      0.94      9197
              1       0.00      0.00      0.00      1104
    
    avg / total       0.80      0.89      0.84     10301
    
    for genre  drama
                 precision    recall  f1-score   support
    
              0       0.56      0.77      0.65      5238
              1       0.60      0.37      0.46      5063
    
    avg / total       0.58      0.57      0.55     10301
    
    for genre  action
                 precision    recall  f1-score   support
    
              0       0.87      1.00      0.93      8959
              1       0.00      0.00      0.00      1342
    
    avg / total       0.76      0.87      0.81     10301
    
    for genre  comedy
                 precision    recall  f1-score   support
    
              0       0.70      1.00      0.82      7200
              1       0.00      0.00      0.00      3101
    
    avg / total       0.49      0.70      0.58     10301
    
    for genre  documentary
                 precision    recall  f1-score   support
    
              0       0.91      1.00      0.95      9359
              1       0.00      0.00      0.00       942
    
    avg / total       0.83      0.91      0.87     10301
    
    for genre  war
                 precision    recall  f1-score   support
    
              0       0.96      1.00      0.98      9860
              1       0.00      0.00      0.00       441
    
    avg / total       0.92      0.96      0.94     10301
    
    


```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
for g in genres:
    clf.fit(train_feature_matrix,train[g])
    y_pred = clf.predict(test_feature_matrix)
    print 'for "%s", f1 score = %.2f' %(g,accuracy_score(test[g],y_pred))
```

    for "sci-fi", classification accuracy = 0.93
    for "horror", classification accuracy = 0.90
    for "fantasy", classification accuracy = 0.95
    for "adventure", classification accuracy = 0.91
    for "thriller", classification accuracy = 0.84
    for "mystery", classification accuracy = 0.94
    for "romance", classification accuracy = 0.85
    for "crime", classification accuracy = 0.89
    for "drama", classification accuracy = 0.58
    for "action", classification accuracy = 0.87
    for "comedy", classification accuracy = 0.70
    for "documentary", classification accuracy = 0.91
    for "war", classification accuracy = 0.96
    


```python

    

```

                 precision    recall  f1-score   support
    
         action       0.38      0.25      0.31      1711
      adventure       0.00      0.00      0.00       593
      animation       0.00      0.00      0.00       247
       children       0.00      0.00      0.00       208
         comedy       0.38      0.47      0.42      3196
          crime       0.00      0.00      0.00       781
    documentary       0.00      0.00      0.00      1144
          drama       0.33      0.70      0.45      3619
        fantasy       0.00      0.00      0.00        53
      film-noir       0.00      0.00      0.00        16
         horror       0.00      0.00      0.00       676
        musical       0.00      0.00      0.00        45
        mystery       0.00      0.00      0.00        71
        romance       0.00      0.00      0.00        68
         sci-fi       0.00      0.00      0.00        85
       thriller       0.00      0.00      0.00       132
            war       0.00      0.00      0.00        12
        western       0.00      0.00      0.00       105
    
    avg / total       0.24      0.35      0.27     12762
    
    


```python


```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
y_pred = logReg.predict(test_feature_matrix)
print classification_report(test['main_genre'],y_pred)
```

                 precision    recall  f1-score   support
    
         action       0.36      0.27      0.31      1711
      adventure       0.15      0.02      0.03       593
      animation       0.50      0.00      0.01       247
       children       0.00      0.00      0.00       208
         comedy       0.37      0.48      0.42      3196
          crime       0.20      0.01      0.02       781
    documentary       0.32      0.01      0.02      1144
          drama       0.33      0.65      0.44      3619
        fantasy       0.00      0.00      0.00        53
      film-noir       0.00      0.00      0.00        16
         horror       0.30      0.06      0.09       676
        musical       0.00      0.00      0.00        45
        mystery       0.00      0.00      0.00        71
        romance       0.00      0.00      0.00        68
         sci-fi       0.00      0.00      0.00        85
       thriller       0.00      0.00      0.00       132
            war       0.00      0.00      0.00        12
        western       0.00      0.00      0.00       105
    
    avg / total       0.31      0.35      0.28     12762
    
    


```python
data2 = pd.read_csv('imdb.csv',quotechar='"',error_bad_lines=False)
```

    Skipping line 66: expected 44 fields, saw 46
    Skipping line 111: expected 44 fields, saw 45
    Skipping line 198: expected 44 fields, saw 45
    Skipping line 222: expected 44 fields, saw 46
    Skipping line 278: expected 44 fields, saw 45
    Skipping line 396: expected 44 fields, saw 45
    Skipping line 403: expected 44 fields, saw 45
    Skipping line 421: expected 44 fields, saw 45
    Skipping line 437: expected 44 fields, saw 45
    Skipping line 462: expected 44 fields, saw 46
    Skipping line 491: expected 44 fields, saw 45
    Skipping line 515: expected 44 fields, saw 45
    Skipping line 529: expected 44 fields, saw 45
    Skipping line 530: expected 44 fields, saw 45
    Skipping line 558: expected 44 fields, saw 45
    Skipping line 623: expected 44 fields, saw 45
    Skipping line 646: expected 44 fields, saw 45
    Skipping line 663: expected 44 fields, saw 46
    Skipping line 713: expected 44 fields, saw 45
    Skipping line 730: expected 44 fields, saw 47
    Skipping line 791: expected 44 fields, saw 45
    Skipping line 813: expected 44 fields, saw 45
    Skipping line 837: expected 44 fields, saw 45
    Skipping line 861: expected 44 fields, saw 45
    Skipping line 874: expected 44 fields, saw 45
    Skipping line 899: expected 44 fields, saw 45
    Skipping line 917: expected 44 fields, saw 45
    Skipping line 944: expected 44 fields, saw 46
    Skipping line 994: expected 44 fields, saw 45
    Skipping line 1027: expected 44 fields, saw 45
    Skipping line 1046: expected 44 fields, saw 45
    Skipping line 1097: expected 44 fields, saw 45
    Skipping line 1106: expected 44 fields, saw 45
    Skipping line 1170: expected 44 fields, saw 45
    Skipping line 1194: expected 44 fields, saw 45
    Skipping line 1195: expected 44 fields, saw 45
    Skipping line 1218: expected 44 fields, saw 45
    Skipping line 1220: expected 44 fields, saw 45
    Skipping line 1270: expected 44 fields, saw 45
    Skipping line 1338: expected 44 fields, saw 47
    Skipping line 1355: expected 44 fields, saw 45
    Skipping line 1363: expected 44 fields, saw 45
    Skipping line 1395: expected 44 fields, saw 45
    Skipping line 1402: expected 44 fields, saw 46
    Skipping line 1418: expected 44 fields, saw 45
    Skipping line 1431: expected 44 fields, saw 45
    Skipping line 1617: expected 44 fields, saw 45
    Skipping line 1663: expected 44 fields, saw 45
    Skipping line 1742: expected 44 fields, saw 46
    Skipping line 1766: expected 44 fields, saw 45
    Skipping line 1799: expected 44 fields, saw 45
    Skipping line 1867: expected 44 fields, saw 45
    Skipping line 1899: expected 44 fields, saw 45
    Skipping line 1900: expected 44 fields, saw 45
    Skipping line 1901: expected 44 fields, saw 45
    Skipping line 1907: expected 44 fields, saw 45
    Skipping line 1913: expected 44 fields, saw 45
    Skipping line 1924: expected 44 fields, saw 45
    Skipping line 1939: expected 44 fields, saw 45
    Skipping line 1945: expected 44 fields, saw 45
    Skipping line 1982: expected 44 fields, saw 45
    Skipping line 2023: expected 44 fields, saw 45
    Skipping line 2028: expected 44 fields, saw 45
    Skipping line 2054: expected 44 fields, saw 45
    Skipping line 2076: expected 44 fields, saw 45
    Skipping line 2081: expected 44 fields, saw 45
    Skipping line 2092: expected 44 fields, saw 45
    Skipping line 2107: expected 44 fields, saw 45
    Skipping line 2160: expected 44 fields, saw 45
    Skipping line 2260: expected 44 fields, saw 45
    Skipping line 2261: expected 44 fields, saw 45
    Skipping line 2289: expected 44 fields, saw 46
    Skipping line 2290: expected 44 fields, saw 45
    Skipping line 2349: expected 44 fields, saw 45
    Skipping line 2395: expected 44 fields, saw 45
    Skipping line 2507: expected 44 fields, saw 45
    Skipping line 2584: expected 44 fields, saw 45
    Skipping line 2588: expected 44 fields, saw 46
    Skipping line 2595: expected 44 fields, saw 45
    Skipping line 2604: expected 44 fields, saw 45
    Skipping line 2622: expected 44 fields, saw 45
    Skipping line 2661: expected 44 fields, saw 45
    Skipping line 2714: expected 44 fields, saw 45
    Skipping line 2722: expected 44 fields, saw 45
    Skipping line 2776: expected 44 fields, saw 45
    Skipping line 2806: expected 44 fields, saw 45
    Skipping line 2826: expected 44 fields, saw 45
    Skipping line 2882: expected 44 fields, saw 45
    Skipping line 2909: expected 44 fields, saw 45
    Skipping line 3005: expected 44 fields, saw 45
    Skipping line 3019: expected 44 fields, saw 45
    Skipping line 3052: expected 44 fields, saw 45
    Skipping line 3062: expected 44 fields, saw 45
    Skipping line 3086: expected 44 fields, saw 45
    Skipping line 3089: expected 44 fields, saw 45
    Skipping line 3134: expected 44 fields, saw 46
    Skipping line 3157: expected 44 fields, saw 45
    Skipping line 3163: expected 44 fields, saw 45
    Skipping line 3177: expected 44 fields, saw 45
    Skipping line 3190: expected 44 fields, saw 45
    Skipping line 3205: expected 44 fields, saw 45
    Skipping line 3209: expected 44 fields, saw 45
    Skipping line 3238: expected 44 fields, saw 45
    Skipping line 3242: expected 44 fields, saw 45
    Skipping line 3255: expected 44 fields, saw 45
    Skipping line 3303: expected 44 fields, saw 45
    Skipping line 3314: expected 44 fields, saw 45
    Skipping line 3322: expected 44 fields, saw 45
    Skipping line 3358: expected 44 fields, saw 45
    Skipping line 3360: expected 44 fields, saw 46
    Skipping line 3377: expected 44 fields, saw 45
    Skipping line 3413: expected 44 fields, saw 45
    Skipping line 3481: expected 44 fields, saw 45
    Skipping line 3496: expected 44 fields, saw 45
    Skipping line 3719: expected 44 fields, saw 45
    Skipping line 3792: expected 44 fields, saw 45
    Skipping line 3807: expected 44 fields, saw 46
    Skipping line 3858: expected 44 fields, saw 45
    Skipping line 3864: expected 44 fields, saw 45
    Skipping line 3902: expected 44 fields, saw 45
    Skipping line 3943: expected 44 fields, saw 45
    Skipping line 3969: expected 44 fields, saw 45
    Skipping line 4024: expected 44 fields, saw 47
    Skipping line 4044: expected 44 fields, saw 45
    Skipping line 4045: expected 44 fields, saw 45
    Skipping line 4112: expected 44 fields, saw 45
    Skipping line 4149: expected 44 fields, saw 45
    Skipping line 4280: expected 44 fields, saw 45
    Skipping line 4282: expected 44 fields, saw 45
    Skipping line 4308: expected 44 fields, saw 45
    Skipping line 4377: expected 44 fields, saw 45
    Skipping line 4390: expected 44 fields, saw 45
    Skipping line 4404: expected 44 fields, saw 45
    Skipping line 4416: expected 44 fields, saw 45
    Skipping line 4423: expected 44 fields, saw 46
    Skipping line 4540: expected 44 fields, saw 45
    Skipping line 4554: expected 44 fields, saw 45
    Skipping line 4556: expected 44 fields, saw 46
    Skipping line 4572: expected 44 fields, saw 45
    Skipping line 4593: expected 44 fields, saw 45
    Skipping line 4614: expected 44 fields, saw 45
    Skipping line 4688: expected 44 fields, saw 45
    Skipping line 4750: expected 44 fields, saw 45
    Skipping line 4764: expected 44 fields, saw 45
    Skipping line 4765: expected 44 fields, saw 45
    Skipping line 4849: expected 44 fields, saw 45
    Skipping line 4865: expected 44 fields, saw 45
    Skipping line 4892: expected 44 fields, saw 45
    Skipping line 4893: expected 44 fields, saw 45
    Skipping line 4897: expected 44 fields, saw 45
    Skipping line 4923: expected 44 fields, saw 45
    Skipping line 4956: expected 44 fields, saw 45
    Skipping line 4957: expected 44 fields, saw 45
    Skipping line 4962: expected 44 fields, saw 45
    Skipping line 4967: expected 44 fields, saw 45
    Skipping line 4971: expected 44 fields, saw 45
    Skipping line 5057: expected 44 fields, saw 45
    Skipping line 5061: expected 44 fields, saw 45
    Skipping line 5097: expected 44 fields, saw 45
    Skipping line 5125: expected 44 fields, saw 45
    Skipping line 5180: expected 44 fields, saw 45
    Skipping line 5207: expected 44 fields, saw 45
    Skipping line 5339: expected 44 fields, saw 45
    Skipping line 5426: expected 44 fields, saw 45
    Skipping line 5474: expected 44 fields, saw 45
    Skipping line 5511: expected 44 fields, saw 45
    Skipping line 5561: expected 44 fields, saw 45
    Skipping line 5563: expected 44 fields, saw 45
    Skipping line 5689: expected 44 fields, saw 45
    Skipping line 5725: expected 44 fields, saw 45
    Skipping line 5759: expected 44 fields, saw 45
    Skipping line 5796: expected 44 fields, saw 45
    Skipping line 5829: expected 44 fields, saw 45
    Skipping line 5854: expected 44 fields, saw 45
    Skipping line 5886: expected 44 fields, saw 45
    Skipping line 5899: expected 44 fields, saw 45
    Skipping line 5901: expected 44 fields, saw 45
    Skipping line 5970: expected 44 fields, saw 45
    Skipping line 5996: expected 44 fields, saw 45
    Skipping line 6085: expected 44 fields, saw 45
    Skipping line 6087: expected 44 fields, saw 45
    Skipping line 6095: expected 44 fields, saw 45
    Skipping line 6096: expected 44 fields, saw 45
    Skipping line 6098: expected 44 fields, saw 45
    Skipping line 6115: expected 44 fields, saw 46
    Skipping line 6158: expected 44 fields, saw 46
    Skipping line 6174: expected 44 fields, saw 45
    Skipping line 6187: expected 44 fields, saw 45
    Skipping line 6218: expected 44 fields, saw 45
    Skipping line 6266: expected 44 fields, saw 45
    Skipping line 6275: expected 44 fields, saw 45
    Skipping line 6279: expected 44 fields, saw 45
    Skipping line 6296: expected 44 fields, saw 45
    Skipping line 6471: expected 44 fields, saw 46
    Skipping line 6494: expected 44 fields, saw 45
    Skipping line 6497: expected 44 fields, saw 45
    Skipping line 6614: expected 44 fields, saw 46
    Skipping line 6714: expected 44 fields, saw 45
    Skipping line 6727: expected 44 fields, saw 45
    Skipping line 6752: expected 44 fields, saw 45
    Skipping line 6763: expected 44 fields, saw 45
    Skipping line 6817: expected 44 fields, saw 45
    Skipping line 6853: expected 44 fields, saw 45
    Skipping line 6904: expected 44 fields, saw 45
    Skipping line 6914: expected 44 fields, saw 45
    Skipping line 6948: expected 44 fields, saw 45
    Skipping line 6969: expected 44 fields, saw 45
    Skipping line 6979: expected 44 fields, saw 45
    Skipping line 7010: expected 44 fields, saw 47
    Skipping line 7024: expected 44 fields, saw 45
    Skipping line 7036: expected 44 fields, saw 45
    Skipping line 7069: expected 44 fields, saw 45
    Skipping line 7146: expected 44 fields, saw 45
    Skipping line 7168: expected 44 fields, saw 45
    Skipping line 7170: expected 44 fields, saw 45
    Skipping line 7317: expected 44 fields, saw 45
    Skipping line 7399: expected 44 fields, saw 45
    Skipping line 7402: expected 44 fields, saw 45
    Skipping line 7496: expected 44 fields, saw 45
    Skipping line 7584: expected 44 fields, saw 45
    Skipping line 7666: expected 44 fields, saw 45
    Skipping line 7690: expected 44 fields, saw 45
    Skipping line 7704: expected 44 fields, saw 47
    Skipping line 7738: expected 44 fields, saw 45
    Skipping line 7773: expected 44 fields, saw 45
    Skipping line 7803: expected 44 fields, saw 45
    Skipping line 7839: expected 44 fields, saw 45
    Skipping line 7850: expected 44 fields, saw 45
    Skipping line 7910: expected 44 fields, saw 45
    Skipping line 7942: expected 44 fields, saw 45
    Skipping line 7959: expected 44 fields, saw 45
    Skipping line 8024: expected 44 fields, saw 45
    Skipping line 8026: expected 44 fields, saw 45
    Skipping line 8028: expected 44 fields, saw 45
    Skipping line 8033: expected 44 fields, saw 45
    Skipping line 8052: expected 44 fields, saw 45
    Skipping line 8129: expected 44 fields, saw 45
    Skipping line 8138: expected 44 fields, saw 45
    Skipping line 8160: expected 44 fields, saw 46
    Skipping line 8244: expected 44 fields, saw 45
    Skipping line 8255: expected 44 fields, saw 45
    Skipping line 8390: expected 44 fields, saw 45
    Skipping line 8400: expected 44 fields, saw 45
    Skipping line 8429: expected 44 fields, saw 45
    Skipping line 8446: expected 44 fields, saw 46
    Skipping line 8565: expected 44 fields, saw 46
    Skipping line 8622: expected 44 fields, saw 46
    Skipping line 8658: expected 44 fields, saw 45
    Skipping line 8742: expected 44 fields, saw 45
    Skipping line 8748: expected 44 fields, saw 45
    Skipping line 8802: expected 44 fields, saw 45
    Skipping line 8844: expected 44 fields, saw 45
    Skipping line 8874: expected 44 fields, saw 45
    Skipping line 8882: expected 44 fields, saw 45
    Skipping line 8885: expected 44 fields, saw 48
    Skipping line 8910: expected 44 fields, saw 45
    Skipping line 8923: expected 44 fields, saw 45
    Skipping line 8947: expected 44 fields, saw 45
    Skipping line 8958: expected 44 fields, saw 45
    Skipping line 9039: expected 44 fields, saw 46
    Skipping line 9090: expected 44 fields, saw 45
    Skipping line 9112: expected 44 fields, saw 45
    Skipping line 9137: expected 44 fields, saw 45
    Skipping line 9201: expected 44 fields, saw 45
    Skipping line 9257: expected 44 fields, saw 45
    Skipping line 9272: expected 44 fields, saw 45
    Skipping line 9390: expected 44 fields, saw 45
    Skipping line 9487: expected 44 fields, saw 45
    Skipping line 9518: expected 44 fields, saw 45
    Skipping line 9554: expected 44 fields, saw 45
    Skipping line 9576: expected 44 fields, saw 45
    Skipping line 9671: expected 44 fields, saw 45
    Skipping line 9690: expected 44 fields, saw 45
    Skipping line 9758: expected 44 fields, saw 45
    Skipping line 9759: expected 44 fields, saw 45
    Skipping line 9767: expected 44 fields, saw 45
    Skipping line 9776: expected 44 fields, saw 45
    Skipping line 9805: expected 44 fields, saw 45
    Skipping line 9834: expected 44 fields, saw 45
    Skipping line 9837: expected 44 fields, saw 45
    Skipping line 9854: expected 44 fields, saw 45
    Skipping line 9890: expected 44 fields, saw 45
    Skipping line 9897: expected 44 fields, saw 45
    Skipping line 9957: expected 44 fields, saw 45
    Skipping line 9979: expected 44 fields, saw 45
    Skipping line 9980: expected 44 fields, saw 45
    Skipping line 10001: expected 44 fields, saw 45
    Skipping line 10002: expected 44 fields, saw 45
    Skipping line 10023: expected 44 fields, saw 45
    Skipping line 10032: expected 44 fields, saw 45
    Skipping line 10051: expected 44 fields, saw 45
    Skipping line 10059: expected 44 fields, saw 46
    Skipping line 10086: expected 44 fields, saw 45
    Skipping line 10102: expected 44 fields, saw 45
    Skipping line 10118: expected 44 fields, saw 45
    Skipping line 10184: expected 44 fields, saw 45
    Skipping line 10199: expected 44 fields, saw 45
    Skipping line 10204: expected 44 fields, saw 45
    Skipping line 10218: expected 44 fields, saw 45
    Skipping line 10224: expected 44 fields, saw 45
    Skipping line 10294: expected 44 fields, saw 45
    Skipping line 10296: expected 44 fields, saw 45
    Skipping line 10331: expected 44 fields, saw 45
    Skipping line 10342: expected 44 fields, saw 45
    Skipping line 10351: expected 44 fields, saw 45
    Skipping line 10414: expected 44 fields, saw 45
    Skipping line 10430: expected 44 fields, saw 45
    Skipping line 10463: expected 44 fields, saw 45
    Skipping line 10478: expected 44 fields, saw 46
    Skipping line 10533: expected 44 fields, saw 45
    Skipping line 10536: expected 44 fields, saw 45
    Skipping line 10539: expected 44 fields, saw 45
    Skipping line 10549: expected 44 fields, saw 45
    Skipping line 10582: expected 44 fields, saw 45
    Skipping line 10588: expected 44 fields, saw 45
    Skipping line 10598: expected 44 fields, saw 45
    Skipping line 10660: expected 44 fields, saw 45
    Skipping line 10733: expected 44 fields, saw 45
    Skipping line 10806: expected 44 fields, saw 45
    Skipping line 10862: expected 44 fields, saw 45
    Skipping line 10905: expected 44 fields, saw 45
    Skipping line 10993: expected 44 fields, saw 45
    Skipping line 11070: expected 44 fields, saw 45
    Skipping line 11084: expected 44 fields, saw 45
    Skipping line 11110: expected 44 fields, saw 45
    Skipping line 11123: expected 44 fields, saw 45
    Skipping line 11128: expected 44 fields, saw 45
    Skipping line 11129: expected 44 fields, saw 45
    Skipping line 11196: expected 44 fields, saw 45
    Skipping line 11210: expected 44 fields, saw 45
    Skipping line 11254: expected 44 fields, saw 45
    Skipping line 11290: expected 44 fields, saw 45
    Skipping line 11365: expected 44 fields, saw 45
    Skipping line 11433: expected 44 fields, saw 45
    Skipping line 11434: expected 44 fields, saw 45
    Skipping line 11469: expected 44 fields, saw 45
    Skipping line 11475: expected 44 fields, saw 45
    Skipping line 11480: expected 44 fields, saw 45
    Skipping line 11513: expected 44 fields, saw 45
    Skipping line 11522: expected 44 fields, saw 45
    Skipping line 11553: expected 44 fields, saw 45
    Skipping line 11610: expected 44 fields, saw 45
    Skipping line 11641: expected 44 fields, saw 45
    Skipping line 11655: expected 44 fields, saw 46
    Skipping line 11689: expected 44 fields, saw 45
    Skipping line 11753: expected 44 fields, saw 45
    Skipping line 11776: expected 44 fields, saw 45
    Skipping line 11797: expected 44 fields, saw 45
    Skipping line 11809: expected 44 fields, saw 45
    Skipping line 11882: expected 44 fields, saw 45
    Skipping line 11915: expected 44 fields, saw 45
    Skipping line 11917: expected 44 fields, saw 46
    Skipping line 11929: expected 44 fields, saw 45
    Skipping line 11956: expected 44 fields, saw 45
    Skipping line 12031: expected 44 fields, saw 45
    Skipping line 12047: expected 44 fields, saw 46
    Skipping line 12160: expected 44 fields, saw 45
    Skipping line 12180: expected 44 fields, saw 45
    Skipping line 12184: expected 44 fields, saw 45
    Skipping line 12224: expected 44 fields, saw 45
    Skipping line 12227: expected 44 fields, saw 45
    Skipping line 12233: expected 44 fields, saw 45
    Skipping line 12251: expected 44 fields, saw 45
    Skipping line 12256: expected 44 fields, saw 45
    Skipping line 12257: expected 44 fields, saw 45
    Skipping line 12259: expected 44 fields, saw 45
    Skipping line 12355: expected 44 fields, saw 45
    Skipping line 12378: expected 44 fields, saw 45
    Skipping line 12398: expected 44 fields, saw 45
    Skipping line 12486: expected 44 fields, saw 45
    Skipping line 12516: expected 44 fields, saw 45
    Skipping line 12588: expected 44 fields, saw 45
    Skipping line 12595: expected 44 fields, saw 45
    Skipping line 12614: expected 44 fields, saw 45
    Skipping line 12642: expected 44 fields, saw 45
    Skipping line 12701: expected 44 fields, saw 45
    Skipping line 12741: expected 44 fields, saw 46
    Skipping line 12771: expected 44 fields, saw 45
    Skipping line 12777: expected 44 fields, saw 45
    Skipping line 12802: expected 44 fields, saw 45
    Skipping line 12892: expected 44 fields, saw 45
    Skipping line 12910: expected 44 fields, saw 47
    Skipping line 12982: expected 44 fields, saw 46
    Skipping line 13024: expected 44 fields, saw 45
    Skipping line 13052: expected 44 fields, saw 45
    Skipping line 13056: expected 44 fields, saw 45
    Skipping line 13158: expected 44 fields, saw 45
    Skipping line 13170: expected 44 fields, saw 45
    Skipping line 13171: expected 44 fields, saw 45
    Skipping line 13186: expected 44 fields, saw 45
    Skipping line 13240: expected 44 fields, saw 45
    Skipping line 13262: expected 44 fields, saw 45
    Skipping line 13374: expected 44 fields, saw 45
    Skipping line 13407: expected 44 fields, saw 45
    Skipping line 13477: expected 44 fields, saw 45
    Skipping line 13540: expected 44 fields, saw 46
    Skipping line 13569: expected 44 fields, saw 45
    Skipping line 13617: expected 44 fields, saw 46
    Skipping line 13651: expected 44 fields, saw 45
    Skipping line 13663: expected 44 fields, saw 45
    Skipping line 13754: expected 44 fields, saw 46
    Skipping line 13775: expected 44 fields, saw 45
    Skipping line 13804: expected 44 fields, saw 45
    Skipping line 13828: expected 44 fields, saw 45
    Skipping line 13865: expected 44 fields, saw 46
    Skipping line 13883: expected 44 fields, saw 45
    Skipping line 13929: expected 44 fields, saw 45
    Skipping line 13948: expected 44 fields, saw 45
    Skipping line 14022: expected 44 fields, saw 45
    Skipping line 14127: expected 44 fields, saw 45
    Skipping line 14148: expected 44 fields, saw 45
    Skipping line 14173: expected 44 fields, saw 45
    Skipping line 14243: expected 44 fields, saw 45
    Skipping line 14254: expected 44 fields, saw 45
    Skipping line 14318: expected 44 fields, saw 45
    Skipping line 14320: expected 44 fields, saw 45
    Skipping line 14354: expected 44 fields, saw 45
    Skipping line 14434: expected 44 fields, saw 45
    Skipping line 14456: expected 44 fields, saw 45
    Skipping line 14479: expected 44 fields, saw 45
    Skipping line 14565: expected 44 fields, saw 45
    Skipping line 14572: expected 44 fields, saw 45
    Skipping line 14616: expected 44 fields, saw 45
    Skipping line 14636: expected 44 fields, saw 45
    Skipping line 14642: expected 44 fields, saw 47
    Skipping line 14644: expected 44 fields, saw 45
    Skipping line 14649: expected 44 fields, saw 45
    Skipping line 14670: expected 44 fields, saw 45
    Skipping line 14688: expected 44 fields, saw 45
    
    


```python
data2.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fn</th>
      <th>tid</th>
      <th>title</th>
      <th>wordsInTitle</th>
      <th>url</th>
      <th>imdbRating</th>
      <th>ratingCount</th>
      <th>duration</th>
      <th>year</th>
      <th>type</th>
      <th>...</th>
      <th>News</th>
      <th>RealityTV</th>
      <th>Romance</th>
      <th>SciFi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>TalkShow</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>titles01/tt0012349</td>
      <td>tt0012349</td>
      <td>Der Vagabund und das Kind (1921)</td>
      <td>der vagabund und das kind</td>
      <td>http://www.imdb.com/title/tt0012349/</td>
      <td>8.4</td>
      <td>40550.0</td>
      <td>3240.0</td>
      <td>1921.0</td>
      <td>video.movie</td>
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
      <th>1</th>
      <td>titles01/tt0015864</td>
      <td>tt0015864</td>
      <td>Goldrausch (1925)</td>
      <td>goldrausch</td>
      <td>http://www.imdb.com/title/tt0015864/</td>
      <td>8.3</td>
      <td>45319.0</td>
      <td>5700.0</td>
      <td>1925.0</td>
      <td>video.movie</td>
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
      <td>titles01/tt0017136</td>
      <td>tt0017136</td>
      <td>Metropolis (1927)</td>
      <td>metropolis</td>
      <td>http://www.imdb.com/title/tt0017136/</td>
      <td>8.4</td>
      <td>81007.0</td>
      <td>9180.0</td>
      <td>1927.0</td>
      <td>video.movie</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>titles01/tt0017925</td>
      <td>tt0017925</td>
      <td>Der General (1926)</td>
      <td>der general</td>
      <td>http://www.imdb.com/title/tt0017925/</td>
      <td>8.3</td>
      <td>37521.0</td>
      <td>6420.0</td>
      <td>1926.0</td>
      <td>video.movie</td>
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
      <th>4</th>
      <td>titles01/tt0021749</td>
      <td>tt0021749</td>
      <td>Lichter der Großstadt (1931)</td>
      <td>lichter der gro stadt</td>
      <td>http://www.imdb.com/title/tt0021749/</td>
      <td>8.7</td>
      <td>70057.0</td>
      <td>5220.0</td>
      <td>1931.0</td>
      <td>video.movie</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 44 columns</p>
</div>




```python

```


```python

```
