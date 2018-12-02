
import torch
import torch.nn as nn
from torch.autograd import Variable


class SoccerRNN(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, n_layers=1):
        super(SoccerRNN, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        output, hidden = self.rnn(input.view(self.seq_len, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
