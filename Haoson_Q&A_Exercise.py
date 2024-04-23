from torch import nn, cuda, argmax, index_select
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset_builder, load_dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, BertForSequenceClassification, BertModel, AutoModelForQuestionAnswering
from tqdm import tqdm
import evaluate

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")

device = "cuda" if cuda.is_available() else "mps"

# Examples
model.to(device)
def answer_my_question(question, context):
    
    # Tokenize question + context
    input = tokenizer(question, 
                        context, 
                        truncation='only_second', 
                        padding=True,
                        return_tensors='pt').to(device)
        
    # Forward Pass
    pred = model(**input)
    start_pos = argmax(pred.start_logits, dim=-1)
    end_pos = argmax(pred.end_logits, dim=-1)
    tokenized_answer = input['input_ids'][0][start_pos: end_pos+1]
    pred_text = tokenizer.decode(tokenized_answer)
    
    return pred_text

# Demo
paragraph=" ".join(["ChatGPT is an artificial intelligence (AI) chatbot developed by OpenAI and released in November 2022.",
                    'The name "ChatGPT" combines "Chat", referring to its chatbot functionality, and "GPT", which stands',
                    "for Generative Pre-trained Transformer, a type of large language model (LLM).[2] ChatGPT is built upon OpenAI's foundational GPT models,",
                    "specifically GPT-3.5 and GPT-4, and has been fine-tuned (an approach to transfer learning) for conversational applications using a combination of supervised and reinforcement learning techniques."])

answer_my_question(question="What is ChatGPT?",
                   context=paragraph)

answer_my_question(question="What can ChatGPT do?",
                   context=paragraph)

answer_my_question(question="How to train a ChatGPT?",
                   context=paragraph)


# ============
# SQuAD
# ============
data = load_dataset("squad")
train_data = data['train']
validation_data = data['validation']

print(train_data['context'][0])
print(train_data['question'][0])
print(train_data['answers'][0])
print(train_data['context'][0][515:525])  # Look 10 chars after answer_start

# ======================
# Demonstrations
# ======================
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")

device = "cuda" if cuda.is_available() else "mps"


# Move to mps
model.eval()
model.to(device)
train_dataloader = DataLoader(train_data, batch_size=5)

# Step 2: Build a training loop
total_training_loss = 0
pred_answer = []
gold_answer = []

for step, sample in enumerate(tqdm(train_dataloader)):
    
    # Prepare Input
    idx = sample['id']
    context = sample['context']
    question = sample['question']
    answers = sample['answers']
    
    
    # Tokenize question + context
    input = tokenizer(question, 
                        context, 
                        truncation='only_second', 
                        padding=True,
                        return_overflowing_tokens=True,
                        return_tensors='pt').to(device)
        
    # Forward Pass
    pred = model(**input)
    
    for sample_id in range(input['input_ids'].shape[0]):
        
        idx_sample = idx[sample_id]
        answer_start = answers['answer_start'][0][sample_id]
        answer_text = answers['text'][0][sample_id]
        
        start_pos = argmax(pred.start_logits[sample_id])
        end_pos = argmax(pred.end_logits[sample_id])
        tokenized_answer = input['input_ids'][sample_id][start_pos: end_pos+1]
        pred_text = tokenizer.decode(tokenized_answer)
        
        # record answer
        pred_answer.append({'id': idx_sample, 'prediction_text': pred_text})
        gold_answer.append({'id': idx_sample, 
                            'answers': {'text': [answer_text.lower()], 
                                        'answer_start': [answer_start.item()]}})
        

# Evaluate Results
metric = evaluate.load("squad")
metric.compute(predictions=pred_answer, references=gold_answer)


# ======================
# A Simple Pipeline
# ======================
class BertQA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.qa_output = nn.Linear(in_features=768, out_features=2, bias=True)
        
    def forward(self, *args, **kwargs):
        bert_output = self.encoder(*args, **kwargs)
        output = self.qa_output(bert_output['last_hidden_state'])
        return {'start_logits': output[:,:,0], 'end_logits': output[:,:,1]}
    
device = "cuda" if cuda.is_available() else "mps"
qa_model = BertQA()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
qa_model.to(device)

# Step 2: Build a training loop
total_training_loss = 0
pred_answer = []
gold_answer = []
train_dataloader = DataLoader(train_data, batch_size=5)

loss_func = nn.CrossEntropyLoss()
optimizer = Adam(params=qa_model.parameters(),
                  lr=3e-5,
                  eps=1e-8)

lr_schedule = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=10,
                                              num_training_steps= int(len(train_data)))


for step, sample in enumerate(tqdm(train_dataloader)):
    
    # Prepare Input
    idx = sample['id']
    context = sample['context']
    question = sample['question']
    answers = sample['answers']
    
    # Tokenize question + context
    input = tokenizer(question, 
                        context, 
                        truncation='only_second', 
                        padding=True,
                        return_tensors='pt').to(device)
    
    # Feed into the model
    pred = qa_model(**input)
    
    label = torch.zeros(pred['start_logits'].shape)
    start_label = label.clone()
    end_label = label.clone()
    
    # Fake 1 hot label
    start_label[:,2] = 1 
    end_label[:,4] = 1
    
    # Calculate loss
    start_loss = loss_func(input=pred['start_logits'], target=start_label.to(device))
    end_loss = loss_func(input=pred['end_logits'], target=end_label.to(device))
    loss = (start_loss + end_loss) / 2
    
    # backword path
    loss.backward()   # calculate gradients
    optimizer.step()   # update parameters
    lr_schedule.step()   # shrink next step sizes
    optimizer.zero_grad()  # reset gradients to prepare for the next iteration
    
    # Report