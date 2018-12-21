
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


# In[5]:


from gensim.corpora.dictionary import Dictionary
from gensim.utils import tokenize


# In[6]:


from IPython.core.debugger import set_trace


# In[7]:


sentence_length = 128
fpath = '../data/quora/train.csv'
batch_size = 32
shuffle = True
num_workers = 4


# In[9]:


class QuoraDataset(Dataset):
    def __init__(self, fpath, transform=None):
        self.fpath = fpath
        self.df = pd.read_csv(fpath)
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


# In[16]:


class FixSentencesLength(object):
    def __init__(self, sentence_length=128, padding=-1):
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
    
class TokToID(object):
    def __init__(self, documents=[]):
        self.dictionary = Dictionary(documents)
            
    def toID(self, sentence):
        try:
            #set_trace()
            return self.dictionary.doc2idx(tokenize(sentence))
        except:
            set_trace()
        
    def __call__(self, item):
        self.item = item
        item[1] = self.toID(item[1])
        item[2] = self.toID(item[2])
        return item
    
class ToTensor:
    def __init__(self):
        pass
    
    def __call__(self, item):
        item[1] = torch.from_numpy(np.array(item[1]))
        item[2] = torch.from_numpy(np.array(item[2]))
        return item


# In[12]:


qdata = QuoraDataset(fpath)


# In[17]:


documents = qdata.df['question1'] + qdata.df['question2'] + [noop_tok]
documents = [tokenize(sentence) for sentence in documents]

transform = transforms.Compose([
    TokToID(documents),
    FixSentencesLength(sentence_length),
    ToTensor()
])
qdata.transform = transform


# In[ ]:


qdata.transform = None


# In[22]:


dataloader = DataLoader(
    dataset=qdata,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers
)


# In[19]:


for i, batch in enumerate(dataloader):
    print(batch)
    break

