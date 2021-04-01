#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


import os
for dirname, _, filenames in os.walk('C:/Users/Administrator/OneDrive/Desktop/rnn'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np


# In[4]:


import codecs
batch_size = 64  
epochs = 62
latent_dim = 256  
num_samples = 10000  
data_path = 'C:/Users/Administrator/OneDrive/Desktop/rnn/rnn.txt'


# In[5]:


input_texts = []
target_texts = []
input_characters = set()
target_characters = set()


# In[6]:


data=pd.read_table(data_path)
data.head()
print(data.columns)


# In[13]:


input_texts


# In[14]:


input_texts=data['new jersey is sometimes quiet during autumn , and it is snowy in april .']
target_texts=data['కొత్త జెర్సీ శరదృతువు సమయంలో కొన్నిసార్లు నిశ్శబ్దంగా ఉంటుంది మరియు ఏప్రిల్‌లో మంచు ఉంటుంది.']
input_texts=input_texts[:]
target_texts=target_texts[:]
target_texts=['\t'+text+'\n' for text in target_texts]


# In[15]:


for text in input_texts:
    for char in text:
        input_characters.add(char)

for text in target_texts:
    for char in text:
        target_characters.add(char)


# In[16]:


input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])


# In[17]:


print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# In[18]:


input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


# In[19]:


encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens),dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')


# In[20]:


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.            
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.    
    decoder_target_data[i, t:, target_token_index[' ']] = 1.


# In[21]:


print(encoder_input_data.shape)
print(decoder_input_data.shape)
print(decoder_target_data.shape)


# In[23]:


encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# In[ ]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit([encoder_input_data, decoder_input_data],
          decoder_target_data,batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


# In[18]:


encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)


# In[19]:


reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


# In[20]:


def decode_sequence(input_seq):

    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))

    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence


# In[21]:


from nltk.translate.bleu_score import sentence_bleu
for a in range(62):
    correct_sen=target_texts[a: a + 1]
    input_seq = encoder_input_data[a: a + 1]
    decoded_sentence = decode_sequence(input_seq)
    reference =  correct_sen
    candidate =  decoded_sentence
    precision = sentence_bleu(reference, candidate)
    print(precision*100)
    print(correct_sen)
    print('Input sentence:', input_texts[a])
    print('Decoded sentence:', decoded_sentence)


# In[ ]:




