import torch 
import torch.nn as nn 



class TimeGradRNN_LH(nn.Module):
    """ TimeGrad RNN on finite history. h_{t-1} = RNN([x_{t-1}, ..., x_{t-L}])"""
    def __init__(self, input_size : int, hidden_size : int, cell = 'LSTM'):
        super().__init__()
        self.in_size = input_size
        self.hidden_size = hidden_size
        if cell == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, bias=True, batch_first=True,
                              dropout=0.0, bidirectional=False)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=1, bias=True, batch_first=True,
                               dropout=0.0, bidirectional=False)
        else:
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, nonlinearity='tanh', bias=True,
                              batch_first=True, dropout=0.0, bidirectional=False)
        
    def forward(self, x_context : torch.Tensor):
        """Return hidden states history [h_1, ..., h_T] of shape (T, hidden_size)
        x_context : (T,N) tensor
        """
        all_h, h_T = self.rnn(x_context) 
        return all_h[:,-1,:]


class TimeGradRNN_FH(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, cell='GRU'):
        super().__init__()
        self.in_size = input_size
        self.hidden_size = hidden_size

        if cell == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, bias=True, batch_first=True,
                              dropout=0.0, bidirectional=False)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=1, bias=True, batch_first=True,
                               dropout=0.0, bidirectional=False)
        else:
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, nonlinearity='tanh', bias=True,
                              batch_first=True, dropout=0.0, bidirectional=False)

    def forward(self, x_context):
        all_h, _ = self.rnn(x_context)  
        return all_h