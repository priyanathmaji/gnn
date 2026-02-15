import torch
import torch.nn as nn
import torch.optim as optim

class TrainerBasicGNN():
    def __init__(self, model, data):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)
        self.data = data

    def train(self):
        data = self.data
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer

        self.optimizer.zero_grad()
        out, h = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask]) #compute loss based on training nodes
        loss.backward() #Gradient
        optimizer.step() #Update parameters

        accuracy = {}
        predicted_classes = torch.argmax(out[data.train_mask], axis=1) # [0.6, 0.2, 0.7, 0.1] -> 2
        target_classes = data.y[data.train_mask]
        accuracy['train'] = torch.mean(torch.where(predicted_classes == target_classes, 1, 0).float())

        predicted_classes = torch.argmax(out, axis = 1)
        target_classes = data.y
        accuracy['val'] = torch.mean(torch.where(predicted_classes == target_classes, 1, 0).float())

        return loss, h, accuracy