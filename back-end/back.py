import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import initializers
from keras.layers import Dropout, Activation, Embedding, Convolution1D, MaxPooling1D, Input, Dense, Merge, BatchNormalization, Flatten, Reshape, Concatenate, concatenate
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
import io
from gensim.parsing.preprocessing import remove_stopwords

def news2Int(news):
    ints = []
    for word in news.split():
        if word in voc2Int:
            ints.append(voc2Int[word])
        else:
            ints.append(voc2Int['<UNK>'])
    return ints

def padNews(news):
    newsPad = news
    if len(newsPad) < dailyLenThreshold:
        for i in range(dailyLenThreshold - len(newsPad)):
            newsPad.append(voc2Int["<PAD>"])
    elif len(newsPad) > dailyLenThreshold:
        newsPad = newsPad[:dailyLenThreshold]
    return newsPad

def cleanText(text):
    text = text.lower()
    text = text.split()
    tmp = []
    for i in text:
        if i in contractions:
            tmp.append(contractions[i])
        else:
            tmp.append(i)
    text = " ".join(tmp)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'b\'', '', text)
    text = re.sub(r',0', '0', text)
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    return remove_stopwords(text)

print("Data cleaning ...")
inFile = pd.read_csv("./Combined_News_DJIA.csv")
price = []
headlines = []
headlineNum = 20
for i in inFile.iterrows():
    headline = []
    price.append(i[1]["Label"])
    for j in range(headlineNum):
        headline.append(i[1]["Top" + str(j + 1)])
    headlines.append(headline)
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

headlinesCleaned = []
for i in headlines:
    headlineCleaned = []
    for j in i:
        headlineCleaned.append(cleanText(j))
    headlinesCleaned.append(headlineCleaned)
wordCnt = {}
for i in headlinesCleaned:
    for j in i:
        for k in j.split():
            if k not in wordCnt:
                wordCnt[k] = 1
            else:
                wordCnt[k] += 1
embedIdx = {}
with io.open("./glove.6B.300d.txt", encoding = "utf-8") as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embed = np.asarray(values[1:], dtype = 'float32')
        embedIdx[word] = embed
wordMiss = 0
threshold = 10
for word, count in wordCnt.items():
    if count > threshold:
        if word not in embedIdx:
            wordMiss += 1
missingRatio = wordMiss * 100. / len(wordCnt)
voc2Int = {} 
value = 0
for word, count in wordCnt.items():
    if count >= threshold or word in embedIdx:
        voc2Int[word] = value
        value += 1
code = ["<UNK>", "<PAD>"]
for i in code:
    voc2Int[i] = len(voc2Int)
int2Voc = {}
for word, value in voc2Int.items():
    int2Voc[value] = word
useRatio = len(voc2Int) * 100. / len(wordCnt)
embedDim = 300
wordNum = len(voc2Int)
embedMatrix = np.zeros((wordNum, embedDim))
for word, i in voc2Int.items():
    if word in embedIdx:
        embedMatrix[i] = embedIdx[word]
    else:
        randEmbed = np.array(np.random.uniform(-1.0, 1.0, embedDim))
        embedIdx[word] = randEmbed
        embedMatrix[i] = randEmbed
kCnt = 0
unkCnt = 0
headlinesInt = []
for i in headlinesCleaned:
    tmp1 = []
    for j in i:
        tmp2 = []
        for k in j.split():
            if k in voc2Int:
                kCnt += 1
                tmp2.append(voc2Int[k])
            else:
                unkCnt += 1
                tmp2.append(voc2Int["<UNK>"])
        tmp1.append(tmp2)
    headlinesInt.append(tmp1)
unk_percent = unkCnt * 100. / (unkCnt + kCnt)
length = []
for i in headlinesInt:
    for j in i:
        length.append(len(j))
length = pd.DataFrame(length, columns = ["# of words"])
headLenThreshold = 15
dailyLenThreshold = 200
headlinesPad = []
for i in headlinesInt:
    dailyHeadlinesPad = []
    for j in i:
        if len(j) <= headLenThreshold:
            for k in j:
                dailyHeadlinesPad.append(k)
        else:
            j = j[:headLenThreshold]
            for k in j:
                dailyHeadlinesPad.append(k)
    if len(dailyHeadlinesPad) < dailyLenThreshold:
        for i in range(dailyLenThreshold - len(dailyHeadlinesPad)):
            dailyHeadlinesPad.append(voc2Int["<PAD>"])
    else:
        dailyHeadlinesPad = dailyHeadlinesPad[:dailyLenThreshold]
    headlinesPad.append(dailyHeadlinesPad)
x_train, x_test, y_train, y_test = train_test_split(headlinesPad, price, test_size = 0.01, random_state = 1)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("Training ...")
filterLen1 = 3
filterLen2 = 5
dropout = 0.1
learningRate = 0.0001
weights = initializers.TruncatedNormal(mean = 0.0, stddev = 0.1, seed = 1)
filterNum = 16
rnnOutputDim = 128
hiddenDim = 128

def build_model():
    model1 = Sequential()
    model1.add(Embedding(wordNum, embedDim, weights = [embedMatrix], input_length = dailyLenThreshold))
    model1.add(Dropout(dropout))
    model1.add(Convolution1D(filters = filterNum, kernel_size = filterLen1, padding = 'same', activation = 'relu'))
    model1.add(Dropout(dropout))
    model1.add(LSTM(rnnOutputDim, activation = None, kernel_initializer = weights, dropout = dropout))
    model2 = Sequential()
    model2.add(Embedding(wordNum, embedDim, weights = [embedMatrix], input_length = dailyLenThreshold))
    model2.add(Dropout(dropout))
    model2.add(Convolution1D(filters = filterNum, kernel_size = filterLen2, padding = 'same', activation = 'relu'))
    model2.add(Dropout(dropout))
    model2.add(LSTM(rnnOutputDim, activation = None, kernel_initializer = weights, dropout = dropout))
    model = Sequential()
    model.add(Merge([model1, model2], mode = 'concat'))
    model.add(Dense(hiddenDim, kernel_initializer = weights))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer = weights, activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = learningRate), metrics = ['accuracy'])
    return model

model = build_model()
save_best_weights = 'question_pairs_weights_lr={}_dropout={}.h5'.format(learningRate, dropout)
callbacks = [ModelCheckpoint(save_best_weights, monitor = 'val_loss', save_best_only = True), EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1, mode = 'auto'), ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, verbose = 1, patience = 3)]
"""
history = model.fit([x_train, x_train], y_train, batch_size = 64, epochs = 100, validation_split = 0.15, verbose = True, shuffle = True, callbacks = callbacks)
#print(history.history.keys())
plt.plot(np.sqrt(np.array(history.history["loss"])), label = "training crossentropy", linewidth = 0.5)
plt.plot(np.sqrt(np.array(history.history["val_loss"])), label = "validation crossentropy", linewidth = 0.5)
plt.plot(np.sqrt(np.array(history.history["acc"])), label = "training accuracy", linewidth = 0.5)
plt.plot(np.sqrt(np.array(history.history["val_acc"])), label = "validation accuracy", linewidth = 0.5)
plt.legend()
plt.ylim(0., 1.)
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig("./learningCurve.pdf")
plt.clf()
"""
print("Predicting ...")
model.load_weights('./question_pairs_weights_lr={}_dropout={}.h5'.format(learningRate,dropout))
y_pred = model.predict([x_test,x_test], verbose = True)
"""
print("true\tpred")
for i in range(len(y_pred)):
    print(str(y_test[i]) + "\t" + str(int(y_pred[i] > 0.5)))
"""

Input = ""
with io.open("../middle-end/middleOut.txt", encoding = "utf-8") as f:
    for line in f:
        Input = u' '.join((Input, line))
cleanInput = cleanText(Input)
intInput = news2Int(cleanInput)
padInput = padNews(intInput)
padInput = np.array(padInput).reshape((1, -1))
output = model.predict([padInput,padInput])
if output[0] > 0.5:
    print("the stock price will rise!")
else:
    print("the stock price will drop!")
