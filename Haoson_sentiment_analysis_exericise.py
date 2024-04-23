from torch import nn
from torch import cuda
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
import torchtext
from torchtext.data import get_tokenizer
from tqdm import tqdm

# ==================
# Load Data
# ==================
train = pd.read_csv('data/imdb_train.csv')
test = pd.read_csv('data/imdb_test.csv')


# Load pre-trained GloVe embeddings
glove = torchtext.vocab.GloVe(name='6B', dim=50)
print(glove.vectors)
print(glove.stoi)   # string to integer
print(glove.vectors.shape)


unk_token = "<unk>"
unk_index = 0
glove_vocab = torchtext.vocab.vocab(glove.stoi)
glove_vocab.insert_token("<unk>",unk_index)
glove_vocab.set_default_index(unk_index)

glove_vocab.lookup_token(0)

# Add an zero vector to glove_vectors
pretrained_embeddings = glove.vectors
pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))

# ==================
# Setup Model
# ==================
word_embedding_dim = 50
hidden_layer_dim = 100
class_dim = 2

# Build a simple Neural Network
class SimpleNeuralClassifier(nn.Module):

    # Constructor: used to initialize a instance of the class
    def __init__(self, pretrained_embeddings):
        super().__init__()  # used to run the constructor of the superclass

        # Define model parameters
        self.embedding_layer = nn.EmbeddingBag.from_pretrained(pretrained_embeddings, 
                                                               freeze=False,
                                                               mode='mean')
        self.hidden_layer = nn.Linear(in_features=word_embedding_dim,
                                      out_features=hidden_layer_dim,
                                      bias=True)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(in_features=hidden_layer_dim,
                                      out_features=class_dim)
        self.softmax = nn.Softmax(dim=-1)

    # Define the forward path of the NN model
    def forward(self, tokens, offsets):
        
        embeddings = self.embedding_layer(tokens, offsets)
        hidden_output_linear = self.hidden_layer(embeddings)
        hidden_activation = self.activation(hidden_output_linear)
        output_linear = self.output_layer(hidden_activation)
        output_prob = self.softmax(output_linear)
        
        return output_prob
    

# ===============
# Train
# ===============

# Step 1:  Set up Tokenizer, Model, Dataloader, 
# Loss, Optimizer, and Scheduler

tokenizer = get_tokenizer("basic_english")

model = SimpleNeuralClassifier(pretrained_embeddings=pretrained_embeddings)

train_dataset = Dataset.from_pandas(train).shuffle(seed=42).with_format('torch')
train_dataloader = DataLoader(train_dataset, batch_size=1)

loss_func = nn.CrossEntropyLoss()
optimizer = Adam(params=model.parameters(),
                  lr=3e-5,
                  eps=1e-8)

lr_schedule = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=10,
                                              num_training_steps= int(len(train_dataset)))

# Step 2: Build a training loop
total_training_loss = 0
for step, sample in enumerate(train_dataloader):

    # Prepare Input
    input = sample['text'][0]
    label = sample['label']
    token_id = torch.tensor(glove_vocab(tokenizer(input)))

    # Forward Pass
    pred = model(tokens = token_id, offsets=torch.tensor([0]))  # offset indicates the position of the first token if token_id is 1D
    loss = loss_func(input=pred,target=label)     # Calculate loss
    
    # Backward Pass
    loss.backward()   # calculate gradients
    optimizer.step()   # update parameters
    lr_schedule.step()   # shrink next step sizes
    optimizer.zero_grad()  # reset gradients to prepare for the next iteration
    
    # Monitor training loss
    total_training_loss += loss.item()
    if step % 1000 == 0 and step != 0:
        print(f"Training Loss at step {step} is {total_training_loss/(step+1)}")
    
    # Ony train on 10000 examples
    if step > 5000:
        break

# See that the embeddings have changed
model.embedding_layer.weight[0]

# ===============
# Evaluate
# ===============
ave_validation_loss = 0
test_dataset = Dataset.from_pandas(test).shuffle(seed=42).with_format('torch')
test_dataloader = DataLoader(test_dataset, batch_size=1)


acc_list = []
for step, sample in enumerate(test_dataloader):
    
    # Prepare Input
    input = sample['text'][0]
    label = sample['label']
    token_id = torch.tensor(glove_vocab(tokenizer(input)))

    # Forward Pass
    pred = model(token_id, offsets=torch.tensor([0]))
    
    # No backward pass
    
    # Monitor validation results
    ave_validation_loss += loss.item()
    pred_outcome = pred[0][1] > pred[0][0]
    gold_label = label == 1
    acc_list.append((pred_outcome == gold_label).item())
    
    if step > 50:
        break

acc = np.mean(acc_list)
print(f'Validation Accurracy is {acc}')


# ==================
# A LLM
# ==================
from transformers import AutoTokenizer, BertForSequenceClassification
bert_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
bert = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", num_labels=2)

device = "cuda" if cuda.is_available() else "mps"

bert.parameters

optimizer = Adam(params=bert.parameters(),
                  lr=3e-5,
                  eps=1e-8)

lr_schedule = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=0,
                                              num_training_steps= int(len(train_dataset)))

# Move to mps
bert.train()
bert.to(device)
train_dataloader = DataLoader(train_dataset, batch_size=5)

# Step 2: Build a training loop
total_training_loss = 0
for step, sample in enumerate(tqdm(train_dataloader)):

    # Prepare Input
    input = sample['text']
    label = sample['label']
    token_id = bert_tokenizer(input, truncation=True, padding=True, return_tensors='pt').to(device)

    # Forward Pass
    pred = bert(**token_id)  # offset indicates the position of the first token if token_id is 1D
    loss = loss_func(input=pred.logits,target=label.to(device))     # Calculate loss
    
    # Backward Pass
    loss.backward()   # calculate gradients
    optimizer.step()   # update parameters
    lr_schedule.step()   # shrink next step sizes
    optimizer.zero_grad()  # reset gradients to prepare for the next iteration
    
    # Monitor training loss
    total_training_loss += loss.item()
    if step % 10 == 0 and step != 0:
        print(f"Training Loss at step {step} is {total_training_loss/(step+1)}")
    
    # Ony train on 10000 examples
    if step > 50:
        break
    
# Evaluate
bert.eval()

ave_validation_loss = 0
test_dataset = Dataset.from_pandas(test).shuffle(seed=42).with_format('torch', device=device)
test_dataloader = DataLoader(test_dataset, batch_size=10)


acc_list = []
for step, sample in enumerate(tqdm(test_dataloader)):

    # Prepare Input
    input = sample['text']
    label = sample['label']
    token_id = bert_tokenizer(input, truncation=True, padding=True, return_tensors='pt').to(device)

    # Forward Pass
    pred = bert(**token_id)  # offset indicates the position of the first token if token_id is 1D
    loss = loss_func(input=pred.logits,target=label.to(device))     # Calculate loss
    
    # No backward pass
    
    # Monitor validation results
    ave_validation_loss += loss.item()
    pred_outcome = pred.logits[:,1] > pred.logits[:,0]
    gold_label = label == 1
    acc_list.extend((pred_outcome == gold_label).tolist())
    
    if step > 50:
        break

acc = np.mean(acc_list)
print(f'Validation Accurracy is {acc}')