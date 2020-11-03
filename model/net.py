"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn import Attention
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        #parameters
        self.attention_size = 512
        # the embedding takes as input the vocab_size and the embedding_dim
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        self.lstm_hidden_dim = params.lstm_hidden_dim
        self.n_layers = params.n_layers
        # the LSTM takes as input the size of its input (embedding_dim), its hidden size
        # for more details on how to use it, check out the documentation
        self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim,\
                    num_layers=params.n_layers, bidirectional=True,\
                    dropout=params.dropout)
        self.dropout = nn.Dropout(params.dropout)
        self.attention = Attention(512)
        # the fully connected layer transforms the output to give the final output layer
        self.fc = nn.Linear(params.lstm_hidden_dim*2, params.number_of_classes)
        self.w_omega = Variable(torch.zeros(params.lstm_hidden_dim * params.n_layers, self.attention_size).cuda())
        self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
       
    def attention_net(self, lstm_output):
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.lstm_hidden_dim*self.n_layers])
        sequence_length = lstm_output.size()[0]
        batch_size = lstm_output.size()[1]
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1,sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        #print(alphas.size()) = (batch_size, squence_length)

        #alphas_reshape = torch.Tensor.reshape(alphas, [-1,sequence_length, 1])
        #print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        alphas_reshape = torch.ones(batch_size, sequence_length, 1).cuda()
        state = lstm_output.permute(1, 0, 2)
        #print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        #print(attn_output.size()) = (batch_size, hidden_size*layer_size)
        return attn_output

    def forward(self, s):

        embedded = self.embedding(s)            # dim: batch_size x seq_len x embedding_dim
        embedded = embedded.permute(1, 0, 2)  # dim: seq_len, batch_size, embedding_dim

        # run the LSTM along the sentences of length seq_len
        output, (hidden_state, cell_state)  = self.lstm(embedded)
        #output dim: seq_len, batch_size, num_directions*hidden_size
        #hidden_state dim: num_layers*num_directions, batch_size, hidden_size
        #-----------------using attention layer---------------------
        attn_output = self.attention_net(output)
        hidden = self.dropout(attn_output)
        #----------------without attention---------------------  
        #hidden = self.dropout(torch.cat((hidden_state[-2], hidden_state[-1]), dim=1))
        #hidden = [batch size, lstm_hidden_dim * num directions]


        return self.fc(hidden) # dim: batch_size x num_tags


def loss_fn(outputs, labels):
    return F.cross_entropy(outputs, labels)
    
    
def accuracy(outputs, labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs==labels)/float(len(labels))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
