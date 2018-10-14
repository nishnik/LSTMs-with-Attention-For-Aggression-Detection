
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import codecs
from nltk.corpus import stopwords
import string
from scipy import sparse
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from math import log
import operator

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.spellcorrect import SpellCorrector

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
import math


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    # annotate={"hashtag", "allcaps", "elongated", "repeated",
    #     'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    # corpus from which the word statistics are going to be used 
    # for spell correction
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def load_train():
    training = pd.read_csv('agr_en_train.csv', encoding='utf-8')
    training = training.fillna('nothing')
    arr = training.as_matrix()
    corpus = arr[:,1]
    y = arr[:,2]
    for i in range(len(y)):
        if (y[i] == 'NAG'):
            y[i] = 0
        elif (y[i] =='OAG'):
            y[i] = 2
        elif (y[i] == 'CAG'):
            y[i] = 1
    corpus = corpus.tolist()        
    y = y.astype(int)
    return corpus, y

def clean_data(corpus):
    new_corp = []
    # spell correct
    for i in range(len(corpus)):
        t = text_processor.pre_process_doc(corpus[i])
        t = [sp.correct(word) for word in t if word not in string.punctuation]
        new_corp.append(" ".join(t))
    new_corp2 = []
    # lemmatize
    for i in range(len(corpus)):
        if (i%1000 == 0):
            print (i) # for logging purpose
        t = new_corp[i].split(" ")
        to_add = []
        for i in t:
            if i not in stop_words:
                to_add.append(wordnet_lemmatizer.lemmatize(i))
        new_corp2.append(" ".join(to_add))
    del new_corp
    return new_corp2



sp = SpellCorrector(corpus="english") 
wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

corpus, y = load_train()
lemmatized_data = clean_data(corpus)

del corpus

all_words = set()
for i in range(len(corpus)):
    print (i)
    t = lemmatized_data[i].split(" ")
    all_words |= set(t)

all_words = list(all_words)

new_corp = lemmatized_data

train_corp = new_corp[:int(len(new_corp)*0.8)]
train_y = y[:int(len(new_corp)*0.8)]
test_corp = new_corp[int(len(new_corp)*0.8):]
test_y = y[int(len(new_corp)*0.8):]



class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)
    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]
        # Here we assume q_dim == k_dim (dot product attention)
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize
        values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size,atten_mod,dropout_p):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
        self.attention = atten_mod
        self.dropout = nn.Dropout(dropout_p)
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        x = self.dropout(x)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        hidden = self.hidden[1]
        hidden = hidden[-1]
        energy, linear_combination = self.attention(hidden, lstm_out, lstm_out) 
        y  = self.hidden2label(linear_combination)
        log_probs = F.log_softmax(y)
        return log_probs

def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
            right += 1.0
    return right/len(truth)


def train_epoch(model, train_data, y, loss_function, optimizer, i):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []
    for sent, label in zip(train_data,y):
        truth_res.append(label)
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        tmp = [all_words.index(w) for w in sent.split(' ')]
        if len(tmp) == 0:
            pred_res.append(pred_res[0])
            continue
        sent = autograd.Variable(torch.LongTensor(tmp))
        label = autograd.Variable(torch.LongTensor([int(label)]))
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        if count % 500 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count, loss.data[0]))
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
    print('epoch: %d done! \n train avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))


def evaluate(model, test_data, test_y, loss_function, name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for sent, label in zip(test_data, test_y):
        truth_res.append(label)
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        tmp = [all_words.index(w) for w in sent.split(' ')]
        if len(tmp) == 0:
            pred_res.append(pred_res[0])
            continue
        sent = autograd.Variable(torch.LongTensor(tmp))
        label = autograd.Variable(torch.LongTensor([int(label)]))
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
    avg_loss /= len(test_data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc ))
    return acc


attention = Attention(100, 100, 100)
model = LSTMClassifier(embedding_dim=100,hidden_dim=100,
                       vocab_size=len(all_words),label_size=3, atten_mod=attention, dropout_p=0.2)

loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
no_up = 0
EPOCH = 10

best_dev_acc = 0.0
for i in range(EPOCH):
    print('epoch: %d start!' % i)
    train_epoch(model, train_corp, train_y, loss_function, optimizer, i)
    print('now best dev acc:',best_dev_acc)
    dev_acc = evaluate(model,test_corp, test_y,loss_function,'dev')
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        # os.system('rm mr_best_model_acc_*.model')
        print('New Best Dev!!!', dev_acc)
        # torch.save(model.state_dict(), 'best_models/mr_best_model_acc_' + str(int(dev_acc*10000)) + '.model')
        no_up = 0
    else:
        no_up += 1
        if no_up >= 2:
            break



