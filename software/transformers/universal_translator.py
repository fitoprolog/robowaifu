import torch
import torch.nn as nn
import torch.optim as optim
from random import randint 

toy_language = "ABC"
char_map = {"A": "X", "B": "Y", "C": "Z"}

# Define the model architecture
class TransformerModel(nn.Module):
    def __init__(self, num_chars, embedding_dim, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=2, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, num_chars)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2) # Change the batch and sequence dimensions
        output = self.transformer(x, x) # Use the transformer to get the output
        output = self.fc(output[-1]) # Use only the last output for classification
        return output

NUM_CHARS = len(toy_language)
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 2
LR = 0.001
NUM_EPOCHS = 1000

model = TransformerModel(NUM_CHARS, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


print(torch.randperm(len(toy_language)))

# Define the training loop
def train(model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        input_seq = torch.tensor([randint(0,2) for char in range(len(toy_language))])
        output_seq = input_seq
        predictions = model(input_seq.unsqueeze(1))
        loss = criterion(predictions, output_seq)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

# Train the model
train(model, optimizer, criterion, NUM_EPOCHS)

# Evaluate the model on a few examples
test_inputs = ["BAC", "CAB", "ACB", "BCA"]
expected_outputs = ["YXZ", "ZXY", "XYZ", "YZX"]
for input_str, expected_output_str in zip(test_inputs, expected_outputs):
    input_seq = torch.tensor([toy_language.index(char) for char in input_str])
    output_seq = model(input_seq.unsqueeze(1)).argmax(dim=1).tolist()
    print(output_seq)
