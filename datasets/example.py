#!/usr/bin/env python3

from TemperatureDataset import TemperatureTrain
from TemperatureDataset import TemperatureVal
from TemperatureDataset import TemperatureUtils
from torch.utils.data import DataLoader
from torch import nn
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import plotly.express as px

# Set the random seed so utils.split() does not return different splits for each training experiment
# (Remove if you want them to return different splits)

np.random.seed(1000)

utils = TemperatureUtils(database='elise')
pids = utils.getPids()
(train_pids, val_pids) = utils.split(pids, train=0.8, val=0.2)

print('The size of the training and validation split is - {train}, {val}'.format(
    train=train_pids.size, val=val_pids.size
))

# Create the datasets from the randomly shuffled
# NOTE - The first time this code is run, it will cache all query data locally.
#        This can take time. Almost 20 mins. Go grab a coffee or something in the meantime!
#        All subsequent calls will get data from the cache (.cache folder) and will be a lot faster.
training_data = TemperatureTrain(database='elise', pids=train_pids, skipNan=True)
val_data = TemperatureTrain(database='elise', pids=val_pids, skipNan=True)

# Print out the size of the training and validation datasets.
#print(len(training_data), len(val_data))

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
    # Pass
    def forward(self, input_sequence, h0=None, c0=None):
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
        if h0 is None:
            h0 = torch.zeros((1, self.hidden_length), requires_grad=True)
        if c0 is None:
            c0 = torch.zeros((1, self.hidden_length), requires_grad=True)
        input_sequence = input_sequence.reshape(1, -1)
        input_sequence = input_sequence.to(torch.float32)
        output, (final_h_state, final_c_state) = self.lstm(input_sequence, (h0, c0))
        output = self.linear1(final_h_state[-1])
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)

        return output, h0.detach().clone(), c0.detach().clone()

# Method to clip gradient.
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

# Helper method to help train the model.
def train_model(model, train_iter, epoch=None, loss_fn=F.l1_loss, logfolder=None):
    # If you want to move the model to cuda from cpu uncomment the below line
    # model.cuda()

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    # Put the model weights into training mode.
    model.train()

    training_loss = torch.zeros(train_iter.length)

    # When we enumerate over each data point in train_iter it will give us 1 data point
    # per participant. This then needs to be broken down into individual segments / days
    # so that each of them can have a distinctive Y value.
    for idx, data in enumerate(train_iter):
        # Extract the X (skin temp) and Y (gestational age + labor onset = predictor) from the data.
        X = data[0]
        Y = data[1]
        laboroffset = data[2]
        pid = data[3]

        current_mini_idx = 0
        old_mini_idx = 0
        current_Y = Y[current_mini_idx]
        prev_Y = current_Y

        """
        We apply transfer learning within each participant, by not resetting the weights
        of the internal layers. If you want to reset the weights of the internal layer
        the reinit h0, c0 = None inside the mini_segment loop below.
        """
        h0 = None
        c0 = None

        print('\n')
        print('ParticipantID - {pid}'.format(pid=pid))

        # Store the loss vs target for each pid.
        loss_buff = []
        target_buff = []

        while current_mini_idx < Y.size()[0]:
            old_mini_idx = current_mini_idx
            prev_Y = current_Y
            while current_Y == prev_Y and current_mini_idx < Y.size()[0]:
                current_Y = Y[current_mini_idx]
                current_mini_idx += 1

            X_mini_segment = X[old_mini_idx : current_mini_idx]
            target = torch.tensor([Y[current_mini_idx-1]])
            loffset = laboroffset[current_mini_idx-1]

            # This can now be passed as a single sequence to model.
            # TODO : Each mini segment can also be length adjusted and stacked to make a single batch for training.
            # Zero out the gradients before from the last pass.
            optim.zero_grad()
            # We can pass hidden state and cell state from previous sequence into the LSTM.
            (prediction, h0, c0) = model(X_mini_segment, h0, c0)
            loss = loss_fn(prediction, target)
            print('G+L Day - {target}, Sequence State - {mini_idx}, Training Loss - {loss}'.
                  format(target=target.item(), mini_idx = current_mini_idx, loss=loss.item()), end='\r', flush=True)
            loss_buff += [loss.detach().numpy()]
            target_buff += [np.abs(loffset)]
            # Chain rule the gradients.
            loss.backward()
            # Clip the gradient so it does not explode.
            clip_gradient(model, 1e-1)
            optim.step()

        if logfolder is not None:
            frame = pd.DataFrame({
                'Days before labor' : target_buff,
                'Training Loss (L1 Loss)' : loss_buff
            })
            frame.to_csv('{logfolder}/loss/train_{pid}.csv'.format(
                logfolder=logfolder,
                pid=pid
            ), index=False)
            fig = px.scatter(frame, x='Days before labor', y='Training Loss (L1 Loss)')
            # Make sure the X Axis is revered, so it can show how the loss is changing.
            fig['layout']['xaxis']['autorange'] = 'reversed'
            fig.write_image('{logfolder}/graphs/train_{pid}.png'.format(logfolder=logfolder, pid=pid))

        training_loss[idx] = loss.item()

    return torch.mean(training_loss), torch.std(training_loss)

# Helper method used to evaluate the model
def eval_model(model, val_iter, epoch=None, loss_fn=F.l1_loss, logfolder=None):
    total_loss = 0
    # Put the model into evaluation mode. So the weights are locked.
    model.eval()

    eval_loss = torch.zeros(val_iter.length)

    # The math operations will not update the gradients.
    with torch.no_grad():
        # The eval loop is similar to the training loop for the mini segments.
        for idx, data in enumerate(val_iter):
            # Extract the X (skin temp) and Y (gestational age + labor onset = predictor) from the data.
            X = data[0]
            Y = data[1]
            laboroffset = data[2]
            pid = data[3]

            current_mini_idx = 0
            old_mini_idx = 0
            current_Y = Y[current_mini_idx]
            prev_Y = current_Y

            """
            We apply transfer learning within each participant, by not resetting the weights
            of the internal layers. If you want to reset the weights of the internal layer
            the reinit h0, c0 = None inside the mini_segment loop below.
            """
            h0 = None
            c0 = None

            print('\n')
            print('ParticipantID - {pid}'.format(pid=pid))

            loss_buff = []
            target_buff = []

            while current_mini_idx < Y.size()[0]:
                old_mini_idx = current_mini_idx
                prev_Y = current_Y
                while current_Y == prev_Y and current_mini_idx < Y.size()[0]:
                    current_Y = Y[current_mini_idx]
                    current_mini_idx += 1

                X_mini_segment = X[old_mini_idx : current_mini_idx]
                target = torch.tensor([Y[current_mini_idx-1]])
                loffset = laboroffset[current_mini_idx-1]

                # We can pass hidden state and cell state from previous sequence into the LSTM.
                (prediction, h0, c0) = model(X_mini_segment, h0, c0)
                loss = loss_fn(prediction, target)
                print('G+L Day - {target}, Sequence State - {mini_idx}, Validation Loss - {loss}'.
                    format(target=target.item(), mini_idx = current_mini_idx, loss=loss.item()), end='\r', flush=True)
                loss_buff += [loss.detach().numpy()]
                target_buff += [np.abs(loffset)]

            if logfolder is not None:
                frame = pd.DataFrame({
                    'Days before labor' : target_buff,
                    'Prediction Loss (L1 Loss)' : loss_buff
                })
                frame.to_csv('{logfolder}/loss/predict_{pid}.csv'.format(
                    logfolder=logfolder,
                    pid=pid
                ), index=False)
                fig = px.scatter(frame, x='Days before labor', y='Prediction Loss (L1 Loss)')
                # Make sure the X Axis is revered, so it can show how the loss is changing.
                fig['layout']['xaxis']['autorange'] = 'reversed'
                fig.write_image('{logfolder}/graphs/predict_{pid}.png'.format(logfolder=logfolder, pid=pid))

            eval_loss[idx] = loss.item()

    return torch.mean(eval_loss), torch.std(eval_loss)

# Start calling the training helper function
model = SimpleLSTM(embed_length=512, hidden_length=256)
# Using a simple L1 loss.
train_loss_avg, train_loss_std = train_model(model, train_iter=training_data, logfolder='../logs_training')
eval_loss_avg, eval_loss_std = eval_model(model, val_iter=val_data, logfolder='../logs_predict')

print('Training Loss : Average - {avg}, Std - {std}'.format(avg=train_loss_avg, std=train_loss_std))
print('Validation Loss : Average - {avg}, Std - {std}'.format(avg=eval_loss_avg, std=eval_loss_std))
