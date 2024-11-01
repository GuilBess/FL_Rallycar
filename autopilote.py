import torch
import torch.nn as nn
import torch.nn.functional as F


import pickle
import lzma


import numpy as np

from PyQt6 import QtWidgets

from data_collector import DataCollectionUI

directions = ["forward", "back", "left", "right"]

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      self.layIn = nn.Linear(16, 9)

      self.fc1 = nn.Linear(9, 9)
      self.fc2 = nn.Linear(9, 9)
      self.fc3 = nn.Linear(9, 9)

      self.out = nn.Linear(9,4)
    
    def forward(self, x):
        x = self.layIn(x)
        x = F.sigmoid(x)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.sigmoid(x)

        out = self.out(x)
        return out
    
    def infer(self, data):
        res = list(torch.round(torch.sigmoid(self(torch.Tensor(data)))))

        out = []

        for idx, i in enumerate(res):
            out.append((directions[idx], int(i)))
        
        return out

    def process_message(self, message, data_collector):
        data = list(message.raycast_distances)
        data.append(message.car_speed)

        commands = self.infer(data)

        for command, start in commands:
            data_collector.onCarControlled(command, start)


model = Net()
model.load_state_dict(torch.load("best_FL_model.pt", weights_only=True))
model.eval()

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    data_window = DataCollectionUI(model.process_message)
    data_window.show()

    app.exec()