# Text Classification

Medical abstracts describe the current conditions of a patient. Doctors routinely scan dozens or hundreds of abstracts each day as they do their rounds in a hospital and must quickly pick up on the salient information pointing to the patient’s malady. 

Design an assistive technology that can identify, with high precision, the class of problems described in the abstract. In the given dataset, abstracts from 5 different conditions have been included: digestive system diseases, cardiovascular diseases, neoplasms, nervous system diseases, and general pathological conditions. 

> Develop predictive models that can determine, given a particular medical abstract,which one of 5 classes it belongs to.

The training dataset consists of 14438 records and the test dataset consists of 14442 records.

### Objectives

- Choose appropriate techniques for modeling text.
- Implement the k-nearest neighbor classifier (cannot use libraries for this algorithm).
- Use your version of the k-NN classifier to assign classes to medical texts.
- Think about dealing with imbalanced data.
- Evaluate results using the F1 Scoring Metric.

### Links

- [Report](report/012457289.pdf)
- [Jupyter Notebook](src/knn_classification_p1.ipynb)

#### Preprocessing - Stemmer fix, Lemmatization, TF-IDF Vectorizer for bag of words with n-grams

#### Unbalanced data - SMOTE over-sampling

### F1-Score = 0.7758



