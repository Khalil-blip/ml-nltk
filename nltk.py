#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk


# In[4]:


nltk.download_shell()


# In[5]:


messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]


# In[6]:


print(len(messages))


# In[7]:


messages[0]


# In[8]:


messages[5000]


# In[9]:


for mess_no,message in enumerate(messages[:10]):
    print(mess_no, message)
    print('\n')


# In[10]:


import pandas as pd


# In[11]:


messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', 
                      names=['label', 'message'])


# In[12]:


messages


# In[13]:


messages.describe()


# In[14]:


messages.groupby('label').describe()


# In[15]:


messages['length'] = messages['message'].apply(len)


# In[16]:


messages.head()


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


import seaborn as sns


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


messages['length'].plot.hist(bins = 100)


# In[21]:


messages['length'].describe()


# In[22]:


messages[messages['length'] == 910]['message'].iloc[0]


# In[23]:


messages.hist(column='length', by='label', bins=60, figsize=(12,4))


# In[24]:


import string


# In[25]:


mess = 'Sample message! notice it has punctuation.'


# In[26]:


string.punctuation


# In[27]:


no_punc = [c for c in mess if c not in string.punctuation]


# In[28]:


no_punc


# In[29]:


from nltk.corpus import stopwords


# In[30]:


stopwords.words('english')


# In[40]:


no_punc = ''.join(no_punc)


# In[41]:


no_punc


# In[42]:


no_punc.split()


# In[43]:


clean_mess = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]


# In[44]:


clean_mess


# In[47]:


def text_process(mess):
    '''
    1. Remove punctuation
    2. remove stopwords
    3. Return list of clean text words
    '''
    
    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[48]:


messages.head()


# In[49]:


messages['message'].head(5).apply(text_process)


# In[50]:


from sklearn.feature_extraction.text import CountVectorizer


# In[51]:


bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])


# In[52]:


print(len(bow_transformer.vocabulary_))


# In[53]:


message4 = messages['message'][3]


# In[54]:


message4


# In[60]:


bow4 = bow_transformer.transform([message4])


# In[61]:


bow4


# In[62]:


print(bow4)


# In[64]:


bow_transformer.get_feature_names()[9554]


# In[76]:


messages_bow = bow_transformer.transform(messages['message'])


# In[77]:


print('Shape of Sparse Matrix: ', messages_bow.shape)


# In[78]:


messages_bow.nnz


# In[79]:


sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


# In[80]:


# Weight and normalization


# In[81]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[93]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[95]:


tfidf4 = tfidf_transformer.transform(bow4)


# In[96]:


print(tfidf4)


# In[105]:


messages_tfidf = tfidf_transformer.transform(messages_bow)
messages_tfidf


# In[98]:


# Training a model


# In[99]:


from sklearn.naive_bayes import MultinomialNB


# In[102]:


spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])


# In[106]:


all_pred = spam_detect_model.predict(messages_tfidf)


# In[107]:


all_pred


# In[111]:


from sklearn.model_selection import train_test_split


# In[113]:


msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)


# In[114]:


msg_train


# In[115]:


from sklearn.pipeline import Pipeline


# In[118]:


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()) 
])


# In[119]:


pipeline.fit(msg_train, label_train) 


# In[120]:


from sklearn.metrics import classification_report


# In[121]:


predictions = pipeline.predict(msg_test)


# In[122]:


print(classification_report(label_test,predictions))


# In[ ]:
Aa

