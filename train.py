import argparse
from dataset import VQADataset
from model import LSTM, Classifier
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import sys
import os

def train(num_epochs, eval_interval, learning_rate, batch_size):
  train_dataset = VQADataset(split='train')
  test_dataset = VQADataset(split='val')
  train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, drop_last=True)
  test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
  criterion = nn.CrossEntropyLoss()

  total_steps = len(train_loader)
  #feature_extractor = torchvision.models.resnet101(pretrained=True).to(device)
  #lstm = LSTM(vocab_size=train_dataset.get_embedding_dim(), embedding_dim=300, hidden_dim=1000).to(device)
  model = Classifier(vocab_size=train_dataset.get_embedding_dim(), embedding_dim=300, hidden_dim=2048, dim_input=2048, dim_output=2048, top_ans=3000).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
  iter = 0
  for epoch in range(num_epochs):
    for i, (images, questions, answers, q_ids, lengths) in enumerate(train_loader):
      images = images.to(device)
      questions = questions.to(device)
      answers = answers.to(device)
      model.train()

      #img_features = feature_extractor(images) # batch_size x 1000
      #word_features = lstm(questions, lengths)
      #combined = img_features * word_features
      #output = model(combined)
      output = model(images, questions, lengths)
      loss = criterion(output, answers)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      if i % 50 == 0:
        curr_iter = epoch * len(train_loader) + i
        print ('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_steps, loss.item()))
        sys.stdout.flush()
      # Do some evaluations
      if iter > 0 and (iter) % eval_interval == 0:
        print('Evaluating at iter {}:'.format(iter))
        curr_acc = evaluate(model, test_loader, train_dataset.inverse_top_answers, iter)
        print('Epoch [{}/{}] Approx. training accuracy: {}'.format(epoch+1, num_epochs, curr_acc))
        if not os.path.exists('models'):
          os.mkdir('models')
        torch.save(model.state_dict(), 'models/model_iter_{}.bin'.format(iter))
        torch.save(optimizer.state_dict(), 'models/optimizer_iter_{}.bin'.format(iter))
      iter += 1

def evaluate(model, test_loader, inverse_dict, iter):
  model.eval()

  predictions = []
  with torch.no_grad():
    for i, (images, questions, q_ids, lengths) in enumerate(test_loader):
      images = images.to(device)
      questions = questions.to(device)
      
      #img_features = feature_extractor(images) # batch_size x 1000
      #word_features = lstm(questions, lengths)
      #combined = img_features * word_features
      #output = model(combined)
      output = model(images, questions, lengths)
      _, answer_index = output.cpu().detach().max(dim=1)

      answer = inverse_dict[answer_index.item()]
      
      answer_dict = {"answer": answer, "question_id": q_ids.item()}
      predictions.append(answer_dict)
      if i % 500 == 0:
        print('Evaluation [{}/{}]'.format(i, len(test_loader)))
        sys.stdout.flush()
    result = json.dumps(predictions)
    if not os.path.exists('results'):
      os.mkdir('results')
    output_file = 'results/val2014_results_iter_{}.json'.format(iter)
    with open(output_file, 'w') as f:
      f.write(result)
    f.close()
    print(output_file)
 
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epochs', '-n', type=int, default=25)
  parser.add_argument('--eval_interval', '-et', type=int, default=5000)
  parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
  parser.add_argument('--batch_size', '-bs', type=int, default=24)
  args = parser.parse_args()
  args = vars(args)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  train(**args)
