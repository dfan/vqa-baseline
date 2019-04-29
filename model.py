import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim):
    super(LSTM, self).__init__()
    self.hidden_dim = hidden_dim
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)

  def forward(self, sentence, q_length):
    sentence = torch.transpose(sentence, 1,0) # from batch x 26 -> 26 x batch
    embeds = self.embedding(sentence)
    packed = embeds
    #packed = pack_padded_sequence(embeds, q_length, batch_first=True)
    _, (_,c) = self.lstm(packed)
    return c.squeeze(0)

class Classifier(nn.Module):
  def __init__(self, dim_input, dim_output, top_ans):
    super(Classifier, self).__init__()
    self.fc = nn.Sequential(
                    nn.Dropout(0.25),
                    nn.Linear(dim_input, dim_output),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(dim_output, top_ans)
                  )
  def forward(self, x):
    return self.fc(x)
