#!/usr/bin/env python3

from TemperatureDataset import TemperatureTrain
from TemperatureDataset import TemperatureVal
from TemperatureDataset import TemperatureUtils
from torch.utils.data import DataLoader
from torch import nn
import torch
import torch.nn.functional as F

import numpy as np

# Set the random seed so utils.split() does not return different splits for each training experiment
# (Remove if you want them to return different splits)

np.random.seed(1000)

utils = TemperatureUtils(database='elise')
pids = utils.getPids()
(train_pids, val_pids) = utils.split(pids, train=0.8, val=0.2)

print('The size of the training and validation split is -')
print(train_pids.size, val_pids.size)

# Create the datasets from the randomly shuffled
# NOTE - The first time this code is run, it will cache all query data locally.
#        This can take time. Almost 20 mins. Go grab a coffee or something in the meantime!
#        All subsequent calls will get data from the cache (.cache folder) and will be a lot faster.
training_data = TemperatureTrain(database='elise', pids=train_pids)
val_data = TemperatureTrain(database='elise', pids=val_pids)

# Print out the size of the training and validation datasets.
print(len(training_data), len(val_data))

class SimpleLSTM(nn.Module):
    """
    Creates an example simple LSTM model that shows how we can use the data.
    Since the sequences are different lengths, I have not batched them yet.
    This will depends on each different application type.
    """
    def __init__(self, embed_length, hidden_length):
        super(SimpleLSTM, self).__init__()
        self.embed_length = embed_length
        self.hidden_length = hidden_length

        # Create the LSTM -> Linear Layer -> .... -> 1 (the combination of gestational age + labor onset)
        # Add any activations that are needed between the linear layers.
        self.lstm = nn.LSTM(embed_length, hidden_length)
        self.linear1 = nn.Linear(hidden_length, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 1)

    # Define the forward pass for the network.
    def forward(self, input_sequence):
        # Set the sequence length to the embedding length.
        # Truncate it if too long or pad it with 0 if small.
        input_length = input_sequence.size()[0]
        if input_length < self.embed_length:
            padding_tensor = torch.zeros(self.embed_length - input_length)
            input_sequence = torch.concat((input_sequence, padding_tensor))
        if input_length > self.embed_length:
            input_sequence = input_sequence[:self.embed_length]

        # Pass the sequence to the LSTM layer.
        # By default the h0 and c0 states are set to 0. We don't care for this example.
        h0 = torch.zeros((1, self.hidden_length), requires_grad=True)
        c0 = torch.zeros((1, self.hidden_length), requires_grad=True)
        input_sequence = input_sequence.reshape(1, -1)
        input_sequence = input_sequence.to(torch.float32)
        output, (final_h_state, final_c_state) = self.lstm(input_sequence, (h0, c0))
        output = self.linear1(final_h_state[-1])
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)

        return output

# Method to clip gradient.
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

# Helper method to help train the model.
def train_model(model, train_iter, epoch=None, loss_fn=F.l1_loss):
    # If you want to move the model to cuda from cpu uncomment the below line
    # model.cuda()

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    # Put the model weights into training mode.
    model.train()

    # When we enumerate over each data point in train_iter it will give us 1 data point
    # per participant. This then needs to be broken down into individual segments / days
    # so that each of them can have a distinctive Y value.

    for idx, data in enumerate(train_iter):
        # Extract the X (skin temp) and Y (gestational age + labor onset = predictor) from the data.
        X = data[0]
        Y = data[1]

        current_mini_idx = 0
        old_mini_idx = 0
        current_Y = Y[current_mini_idx]
        prev_Y = current_Y

        while current_mini_idx < Y.size()[0]:
            old_mini_idx = current_mini_idx
            #print(current_mini_idx)
            prev_Y = current_Y
            while current_Y == prev_Y and current_mini_idx < Y.size()[0]:
                current_Y = Y[current_mini_idx]
                current_mini_idx += 1

            X_mini_segment = X[old_mini_idx : current_mini_idx]
            target = torch.tensor([Y[current_mini_idx-1]])

            # This can now be passed as a single sequence to model.
            # TODO : Each mini segment can also be length adjusted and stacked to make a single batch for training.
            # Zero out the gradients before from the last pass.
            optim.zero_grad()
            prediction = model(X_mini_segment)
            loss = loss_fn(prediction, target)
            print('Data Point - {idx}, Sequence State - {mini_idx}, Loss - {loss}'.
                  format(idx=idx, mini_idx = current_mini_idx, loss=loss.item()), end='\r', flush=True)
            # Chain rule the gradients.
            loss.backward()
            # Clip the gradient so it does not explode.
            clip_gradient(model, 1e-1)
            optim.step()

    return loss.item()

# Start calling the training helper function
model = SimpleLSTM(embed_length=512, hidden_length=256)
# Using a simple L1 loss.
train_loss = train_model(model, train_iter=training_data)
