import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision

# LSTM(vocab_size=train_dataset.get_embedding_dim(), embedding_dim=300, hidden_dim=2048)
class LSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim):
    super(LSTM, self).__init__()
    self.hidden_dim = hidden_dim
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)

  def forward(self, sentence, q_length):
    sentence = torch.transpose(sentence, 1,0) # from batch x 26 -> 26 x batch
    embeds = self.embedding(sentence)
    packed = embeds
    #packed = pack_padded_sequence(embeds, q_length, batch_first=True)
    _, (_,c) = self.lstm(packed)
    return c.squeeze(0)

class Classifier(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, dim_input, dim_output, top_ans):
    super(Classifier, self).__init__()
    self.lstm = LSTM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    self.fc = nn.Sequential(
                    nn.Dropout(0.25),
                    nn.Linear(dim_input, dim_output),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(dim_output, top_ans)
                  )
  def forward(self, images, questions, lengths):
    x = images
    y = self.lstm(questions, lengths)
    z = x+y
    return self.fc(z)
