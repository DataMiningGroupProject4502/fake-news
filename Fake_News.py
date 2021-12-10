# Python Libraries 
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import keras 
import tensorflow 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
import sklearn
from sklearn.model_selection import train_test_split
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# Load Data
true = pd.read_csv('True.csv', low_memory = False)
fake = pd.read_csv('Fake.csv', low_memory = False)
file1 = pd.read_csv('File1.csv', low_memory = False, encoding = 'ISO-8859-1')                         
file2 = pd.read_csv('File2.csv', low_memory = False, encoding = 'ISO-8859-1')                        
file3 = pd.read_csv('File3.csv', low_memory = False, encoding = 'ISO-8859-1')      
file4 = pd.read_csv('File4.csv', low_memory = False, encoding = 'ISO-8859-1')

# Data Preprocessing Steps 
# 1. Data Cleaning
# Fix categorical data 
true = true.iloc[:, [0, 1]]
true['label'] = 0  
true = true.iloc[:, [0, 1, 2]]

fake = fake.iloc[:, [0, 1]] 
fake['label'] = 1    
fake = fake.iloc[:, [0, 1, 2]]

file1 = file1.iloc[:, [0, 1, 2]]
file1 = file1[pd.to_numeric(file1['label'], errors='coerce').notnull()]

file2 = file2.iloc[:, [0, 1, 2]]
file2 = file2.replace(to_replace = ['REAL', 'FAKE'], value = [0, 1])
file2 = file2[pd.to_numeric(file2['label'], errors = 'coerce').notnull()]

file3 = file3.iloc[:, [0, 1, 2]]
file3 = file3[pd.to_numeric(file3['label'], errors = 'coerce').notnull()]

file4 = file4.iloc[:, [0, 1, 2]]
file4 = file4[pd.to_numeric(file4['label'], errors = 'coerce').notnull()]

# 2. Data Integration
# Create and clean master dataset
dataset = pd.concat([true, fake]).reset_index(drop = True)                             
dataset['contents'] = dataset['title'] + ' ' + dataset['text']
dataset = dataset.replace(to_replace = ['0', '1'], value = [0, 1])
dataset.dropna(inplace = True)                                                     

# 3. Data Transformation
# Removing stopwords
nltk.download("stopwords")
stop_words = stopwords.words("english")

def preprocess(text):
    clean_db = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stop_words:
            clean_db.append(token) 
    return clean_db
dataset['clean_contents'] = dataset['contents'].apply(preprocess)

# Total unique words
unique_words = []
for wrd in dataset.clean_contents:
    unique_words.extend(wrd)
total_unique_words = len(list(set(unique_words)))   

# Convert lists of words into strings
dataset['clean_contents'] = dataset['clean_contents'].apply(lambda x: ' '.join(x))

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(dataset.clean_contents, dataset.label, test_size = 0.3)

# Tokenization 
tokenizer = Tokenizer(num_words = total_unique_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

# 4. Data Reduction
padded_train = pad_sequences(train_sequences, maxlen = 50)
padded_test = pad_sequences(test_sequences, maxlen = 50) 

# Train Model
# Sequential Model
model = Sequential()
model.add(Embedding(total_unique_words, output_dim = 128))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
y_train = np.asarray(y_train).astype(np.float32)
model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 1)
pred = model.predict(padded_test)

prediction = [  ]
for i in range(len(pred)):
    if pred[i].item == 0:
        prediction.append(0)
    else if pred[i].item() > 0.5:
        prediction.append(1)

# Calculate Accuracy
y_test = list(y_test)
matches = 0
length = len(prediction)
for i in range(length):
    if y_test[i] == prediction[i]:
        matches += 1
accuracy = matches / length

print("The total number of unique words: ", total_unique_words)
print("Accuracy result: ", accuracy)

'''
# Consufion Metrix Graph based on True and Fake news files
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, prediction)
labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cf_matrix, annot = labels, fmt = '', cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])
plt.show()
'''