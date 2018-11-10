
# coding: utf-8

# In[183]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE


# In[184]:


def give_me_a_frame(data_location):
    return pd.read_csv(
        filepath_or_buffer=data_location, 
        header=None, 
        sep='\n')


# In[185]:


def preprocess(statements):
#     print(statement)
    s = set(stopwords.words('english'))
    stemmer = WordNetLemmatizer()
    table = str.maketrans('', '', string.punctuation)
    digit_table = str.maketrans('', '', string.digits)
    
    kal = []
    for statement in statements:
#         print(statement)
        statement = statement.lower()
        aaj = list(filter(lambda w: not w in s,statement.split()))
        stripped = [w.translate(table) for w in aaj]
        stripped = [w.translate(digit_table) for w in stripped]
        stripped = [d for d in stripped if len(d) > 1]

        stripped_stemmer_fix = [stemmer.lemmatize(stemmer_word) for stemmer_word in stripped]
        kal.append(" ".join(stripped_stemmer_fix))
#     print(stripped)
#     print(kal[:2])
    return kal


# In[186]:


def build_matrix(train, test, n):
    vectorizer = TfidfVectorizer(norm='l2'
                                 ,ngram_range=(1,n)
                                 ,min_df=3 
                                )
    return vectorizer.fit_transform(train),vectorizer.transform(test)


# In[187]:


def predict_output(train_matrix, test_vector, train_classes,  k=3 ):
    dot_product = test_vector.dot(train_matrix.T)
    sims = list(zip(dot_product.indices, dot_product.data))
    sims.sort(key=lambda x: x[1], reverse=True)
#     tc = Counter(train_classes[s[0]] for s in sims[:k]).most_common()
#     if len(tc) < 2 or tc[0][1] > tc[1][1]:
#             # majority vote
#             return tc[0][0]
#     print(train_classes[4863:4867])
    tc = defaultdict(float)
    for s in sims[:k]:
                tc[train_classes[s[0]]] += s[1]
    return sorted(tc.items(), key=lambda x: x[1], reverse=True)[0][0]


# In[174]:


def splitData(mat, cls, fold=1, d=10):
    r""" Split the matrix and class info into train and test data using d-fold hold-out
    """
    n = mat.shape[0]
    r = int(np.ceil(n*1.0/d))
    print(r)
    mattr = []
    clstr = []
    # split mat and cls into d folds
    for f in range(d):
#         if f+1 != fold:
            mattr.append( mat[f*r: min((f+1)*r, n)] )
            clstr.extend( cls[f*r: min((f+1)*r, n)] )
#     print(mattr)
    # join all fold matrices that are not the test matrix
    train = sp.vstack(mattr)
    # extract the test matrix and class values associated with the test rows
#     test = mat[(fold-1)*r: min(fold*r, n), :]
#     clste = cls[(fold-1)*r: min(fold*r, n)]

    return train, clstr


# In[175]:


def classify(statements, statements_test, classes, c, k):
    print("c = ", c)
    print("k = ", k)
    statements = preprocess(statements)
    statements_test = preprocess(statements_test)
    classes = np.array(classes)
    X_Train, X_test = build_matrix(statements,statements_test,c)
    nm = SMOTE(random_state=21)
    X_res, y_res = nm.fit_sample(X_Train, classes)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.33, random_state=21)
    y_pred = [predict_output(X_train,x,y_train,k) for x in X_test]
    print(precision_recall_fscore_support(y_test, y_pred, average='macro'))    
    


# In[188]:


df = give_me_a_frame('train.dat')   
df = pd.DataFrame(df[0].str.split('\t', 1).tolist())
statements = df[1]
classes = df[0]
print(statements[0])

# n, bins, patches = plt.hist(classes)
# plt.show()

df = give_me_a_frame('test.dat')   
statements_test = df[0]
# print(statements_test)
# df = pd.DataFrame({'aaj':statements})

# df[df['aaj'].str.contains('acute')]


# In[189]:


statements = preprocess(statements)
statements_test = preprocess(statements_test)
classes = np.array(classes)
# df = pd.DataFrame({'aaj':statements})

print(statements[:1])
# df[df['aaj'].str.contains('acut')]


X_Train, X_test = build_matrix(statements,statements_test,2)


# In[190]:


nm = SMOTE(random_state=21)
X_res, y_res = nm.fit_sample(X_Train, classes)
labels, values = zip(*Counter(y_res).most_common())
indexes = np.arange(len(labels))
width = 1
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()


# In[ ]:


X_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.33, random_state=21)
# X_train, x_test, y_train, y_test = train_test_split(X_Train, classes, test_size=0.33, random_state=21)

# # X_train.append(x_test)
X_train = sp.vstack([X_train,x_test])
# # y_train.extend(y_test)
y_train = np.append(y_train,y_test)
# # y_pred = [predict_output(X_train,x,y_train,8) for x in X_test]
# print(X_train.shape[0])
# print(y_train.shape[0])

with open('out.dat', 'w') as f:
    for i in range(X_test.shape[0]):
        f.write("%s\n" % (predict_output(X_train, X_test[i], y_train, 73)))
        
# clspr = [predict_output(X_Train, x_test, classes, 5) for x_test in X_test]
# print(clspr)
# aaj = cosine_similarity(X_test[:1],X_Train).flatten()
# kal = X_test[:1].dot(X_Train.T)
# similar_docs = aaj.argsort()[:-6:-1]
# statements = np.array(statements)
# classes = np.array(classes)
# print(similar_docs)
# print(classes[similar_docs])
# print(kal)


# In[182]:


# for c in range(2,3):
#     for k in range(73,74):
#         classify(statements, statements_test, classes, c, k)

