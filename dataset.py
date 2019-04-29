import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from vqaTools.vqa import VQA
import os
import random
from PIL import Image
import numpy as np
import string
from collections import Counter
import itertools
from itertools import takewhile
import json

class VQADataset(data.dataset.Dataset):
  def __init__(self, split):
    dataDir   ='data'
    versionType ='v2_' # this should be '' when using VQA v2.0 dataset
    taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
    dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
    if 'train' in split:
      dataSubType = 'train2014'
    elif 'val' in split:
      dataSubType = 'val2014'
    annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
    quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
    imgDir    = '%s/Images/%s/' %(dataDir, dataSubType)
    
    self.dataSubType = dataSubType
    self.split = split
    self.imgDir = imgDir

    # Initialize VQA API
    vqa = VQA(annFile, quesFile)
    self.vqa = vqa
    
    img_ids = vqa.getImgIds() # get all
    self.img_ids = img_ids
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    self.transform = transforms.Compose([
      transforms.Resize((224,224)), # ImageNet standard
      transforms.ToTensor(),
      transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Create vocabulary mapping letters to numbers
    self.all_letters = string.ascii_letters

    # Get top 3000 answers
    with open(annFile, 'r') as f:
      train_ann = json.load(f)
    all_answers = []
    for list in train_ann['annotations']:
      for answer in list['answers']:
        all_answers.append(answer['answer'])
    if self.split == 'train':
      print('Computing top K answers')
      top_answers = self.get_top_k_answers(all_answers, 3000)
      self.top_answers = top_answers

  def get_embedding_dim(self):
    return len(self.all_letters) + 1
  
  def letterToIndex(self, letter):
    # 0 does not represent a word
    return self.all_letters.find(letter) + 1
  
  def tokenize_answer(self, answer):
    answer_list = []
    for answer in answer['answers']:
      answer_list.append(answer['answer'])
    return answer_list
  
  def encode_text(self, text):
    text_vec = torch.zeros(26).long()
    length = min(len(text), 26)
    for i in range(length):
      token = text[i]
      index = self.letterToIndex(token)
      text_vec[i] = index
    return text_vec, max(length, 1)

  def get_top_k_answers(self, answers, top_k):
    counter = Counter(answers)
    counted_ans = counter.most_common(top_k)
    vocab = {t[0]: i for i, t in enumerate(counted_ans, start=0)}

    return vocab

  def __getitem__(self, index):
    has_answered = False
    while not has_answered:
      img_id = self.img_ids[index]
      question_ids = self.vqa.getQuesIds(imgIds=img_id)
      question_id = random.sample(question_ids, 1)[0]
      answer_dict = self.vqa.loadQA(question_id)[0]
      imgFilename = 'COCO_' + self.dataSubType + '_'+ str(img_id).zfill(12) + '.jpg'
      imgFilename = os.path.join(self.imgDir, imgFilename)
      color = Image.open(imgFilename).convert('RGB')
      color = self.transform(color)
    
      nontoken_answers = self.tokenize_answer(answer_dict)
      occurence_count = Counter(nontoken_answers)
      best_answer = occurence_count.most_common(1)[0][0]
      if self.split == 'val' or best_answer in self.top_answers:
        has_answered = True
      index = (index + 1) % self.__len__()

    question = self.vqa.qqa[question_id]['question']
    question,length = self.encode_text(question)

    if self.split == 'train':
      # best_answer, _ = self.encode_text(best_answer)
      answer_label = self.top_answers[best_answer]
      return color, question, answer_label, question_id, length

    return color, question, question_id, length
    
  def __len__(self):
    return len(self.img_ids)

if __name__ == '__main__':
  dataset = VQADataset(split='train')
  dataset.__getitem__(5)
  dataset.__getitem__(10)
  dataset.__getitem__(15)
