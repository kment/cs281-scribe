import numpy as np
import pickle
import re
import torch
from torch.autograd import Variable

# Read in training data
f_in = open("speeches_concat.txt", 'r')
text = f_in.read()
f_in.close()

# Keep letters and a few symbols only
text = re.sub("[^a-zA-Z .:?\[\]]+", '', text.replace("\n", " ")).lower()

# Settings
n_epochs = 500
n_hidden = 200
seq_length = 30
n_seq = 10000

letters = "".join(np.unique([letter for letter in text]))
n_letters = len(letters)

# Draw n_seq sequences of length seq_length from text
def draw_sequences(text, n_seq, seq_length):

    i_batches = np.random.choice(len(text) - seq_length - 1, n_seq)
    draw = Variable(torch.zeros(seq_length, n_seq, n_letters), requires_grad=False)
    nextchar = Variable(torch.LongTensor(n_seq), requires_grad=False)
    for i in range(n_seq):
        for j in range(seq_length):
            draw.data[j][i][letters.find(text[i_batches[i] + j])] = 1
        nextchar.data[i] = letters.find(text[i_batches[i] + seq_length])
    
    return draw, nextchar

# Train the RNN and return the loss
def train(model, optimizer, loss, data, target):
    
    optimizer.zero_grad()
    output, hc = model.lstm(data)
    new_loss = loss(model.encoder(output[-1]), target)
    new_loss.backward()
    optimizer.step()
    return new_loss.data[0]

# Set up the model and optimizer
model = torch.nn.Sequential()
model.add_module("lstm", torch.nn.LSTM(n_letters, n_hidden, 2))
model.add_module("encoder", torch.nn.Linear(n_hidden, n_letters))
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
costs = np.zeros((2, n_epochs))

# Prepare training and validation data
data_val, value_val = draw_sequences(text, 500, seq_length)
data, value = draw_sequences(text, n_seq, seq_length)

# Train the model
for epoch in range(n_epochs):
    
    costs[0][epoch] = train(model, optimizer, loss, data, value)
    if epoch == 0 or (epoch + 1) % 10 == 0:
        costs[1][epoch] = loss(model.encoder(model.lstm(data_val)[0][-1]), value_val).data[0]
    print("Epoch:\t", epoch + 1, "\tLoss (train):\t", costs[0][epoch], "\tLoss (val):\t", costs[1][epoch])
    
    if (epoch + 1) % 25 == 0:
        with open("modelE" + str(epoch + 1) + ".p", 'wb') as f_model:
            pickle.dump(model, f_model, pickle.HIGHEST_PROTOCOL)

# Sample the trained model: choose an initial sequence
new_text = "Trump:"
new_data = Variable(torch.zeros((1, 1, n_letters)))
for i, letter in enumerate(new_text):
    new_data.data.zero_()
    new_data.data[0][0][letters.find(letter)] = 1
    if i:
        output_, hc_ = model.lstm(new_data, hc_)
    else:
        output_, hc_ = model.lstm(new_data)

# Sample the next 140 characters
for i in range(140):
    new_text += letters[model.encoder(output_).data.topk(1)[1][0][0][0]]
    new_data.data.zero_()
    new_data.data[0][0][model.encoder(output_).data.topk(1)[1][0][0][0]] = 1
    output_, hc_ = model.lstm(new_data, hc_)

print(new_text)