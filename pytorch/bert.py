#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


# In[3]:


from gensim.corpora.dictionary import Dictionary
from gensim.utils import tokenize


# In[4]:


from pytorch_pretrained_bert import BertTokenizer, BertModel


# In[5]:


from IPython.core.debugger import set_trace


# In[6]:


from time import time


# In[7]:


sentence_length = 18
fpath = '../data/quora/train.csv'
batch_size = 32
shuffle = True
num_workers = 8
learning_rate = 1e-3


# In[8]:


class QuoraDataset(Dataset):
    def __init__(self, fpath, transform=None):
        self.fpath = fpath
        self.df = pd.read_csv(fpath).sample(frac=1.0)
        self.df = self.df.dropna()
        self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        item = self.df.iloc[index]
        sample = [item['is_duplicate'], item['question1'], item['question2']]
        if self.transform:
            sample = self.transform(sample)
        return sample        


# In[9]:


class PrepareSentencePair:
    def __init__(self):
        pass
    
    def __call__(self, item):
        assert len(item) == 3
        label = item[0]
        sentence_pair = '%s %s'%(item[1], item[2])
        return [label, sentence_pair]

class FixSentencesLength(object):
    def __init__(self, sentence_length=128, padding=0):
        self.sentence_length = sentence_length
        self.padding = padding
        
    def fix_length(self, sentence_ids):
        if len(sentence_ids) >= self.sentence_length:
            return sentence_ids[:self.sentence_length]
        else:
            return sentence_ids + [self.padding]*(self.sentence_length - len(sentence_ids))
         
    def __call__(self, item):
        assert len(item) == 3
        item[1] = self.fix_length(item[1])
        item[2] = self.fix_length(item[2])
        return item
    
class Concat(object):
    def __init__(self):
        pass
    
    def __call__(self, item):
        concated_ = item[1] + item[2]
        return [concated_, item[0]]
    
class TokToID(object):
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer
            
    def toID(self, sentence):
        try:
            #set_trace()
            toks = self.tokenizer.tokenize(sentence)
            ids = self.tokenizer.convert_tokens_to_ids(toks)
            return ids
        except:
            set_trace()
        
    def __call__(self, item):
        item[1] = self.toID(item[1])
        item[2] = self.toID(item[2])
        return item
    
class ToTensor:
    def __init__(self):
        pass
    
    def __call__(self, item):
        item[0] = torch.from_numpy(np.array(item[0]))
        item[1] = torch.from_numpy(np.array(item[1]))
                
        return item


# In[10]:


class BertSentenceDifference(torch.nn.Module):
    def __init__(self):
        self.global_step = 0
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        for p in self.bert_model.parameters():
            p.requires_grad = False
        
        self.fc1 = torch.nn.Linear(in_features=768, out_features=768, bias=True)
        #self.fc2 = torch.nn.Linear(in_features=768, out_features=2, bias=True)
        self.fc2 = torch.nn.Linear(in_features=768*2, out_features=2, bias=True)
        #self.fc2 = torch.nn.Linear(in_features=768*12, out_features=2, bias=True)
        self.fc1_act = torch.nn.Tanh()
        self.fc2_act = torch.nn.Softmax()
        
    def reshape_bert_embedding(self, bert_output):
        encoder = bert_output[0]
        concat_encoder = torch.stack(encoder).permute(1, 0, 2, 3)[:, :, -1, :]
        return torch.reshape(concat_encoder, (-1, 768*12))
    
    def take_sentence_mean(self, bert_output):
        last_encoding_layer = bert_output[0][-1]
        return torch.mean(last_encoding_layer, 1)
        
    def forward(self, X):
        self.global_step += 1
        
        try:
            sent1, sent2 = X[:, :sentence_length], X[:, sentence_length:]
            segment_tensor = torch.zeros_like(sent1)
            #self.bert_embeddings = [self.bert_model(sent1, segment_tensor)[1], self.bert_model(sent2, segment_tensor)[1]]
            #self.bert_embeddings = [self.reshape_bert_embedding(self.bert_model(sent1, segment_tensor)), self.reshape_bert_embedding(self.bert_model(sent2, segment_tensor))]
            self.bert_embeddings = [self.take_sentence_mean(self.bert_model(sent1, segment_tensor)), self.take_sentence_mean(self.bert_model(sent2, segment_tensor))]

            #self.diff = torch.abs(self.bert_embeddings[0] - self.bert_embeddings[1]).exp()
            #self.diff = self.bert_embeddings[0] - self.bert_embeddings[1]
            #self.diff = torch.mul(self.diff, self.diff)
            #self.diff = self.bert_embeddings[0] * self.bert_embeddings[1]
            self.diff = torch.cat(self.bert_embeddings, dim=1)
            
            #logits = self.fc1(self.diff)
            #logits = self.fc1_act(logits)
            
            logits = self.fc2(self.diff)
            logits = self.fc2_act(logits)

            return logits
        except:
            #set_trace()
            raise


# In[11]:


qdata = QuoraDataset(fpath)
transform = transforms.Compose([
    TokToID(),
    FixSentencesLength(sentence_length),
    Concat(),
    ToTensor()
])
qdata.transform = transform
dataloader = DataLoader(
    dataset=qdata,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers
)


# In[ ]:


from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
model = BertSentenceDifference()
loss = CrossEntropyLoss()
trainable_tensors = [p[1] for p in model.named_parameters() if p[0].startswith('fc2')]
optimizer = Adam(params=trainable_tensors, lr=learning_rate)

from ignite.engine import create_supervised_trainer, Events, create_supervised_evaluator
from ignite.metrics import Loss, Precision
trainer = create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=loss)
evaluator = create_supervised_evaluator(model=model, metrics={'Loss': Loss(loss), 'precision': Precision()})

epoch_st = time()
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(trainer):
    evaluator.run(dataloader)
    global epoch_st
    elasped_time = int(time()-epoch_st)
    epoch_st = time()
    print(f"epoch {trainer.state.epoch} {evaluator.state.metrics} {elasped_time}")
    
trainer.run(dataloader, max_epochs=100)


