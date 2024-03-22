import PySimpleGUI as sg
from ServerComm import *
from ClientComm import *
from Server import *
from dataset.dataset import *
from arch.ResNet18 import *
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from FedAvg import *
from Client import *
from arch.arch import *

from utils.logger import *
from utils.profiler import *

logger().set_log_type(logger_log_type.logger_type_debug.value |
                    logger_log_type.logger_type_error.value |
                    logger_log_type.logger_type_info.value |
                    logger_log_type.logger_type_normal.value |
                    logger_log_type.logger_type_warning.value)

logger().set_stdout(logger_stdout_type.logger_stdout_console.value |
                  logger_stdout_type.logger_stdout_file.value |
                  logger_stdout_type.logger_stdout_network.value |
                  logger_stdout_type.logger_stdout_stringio.value)

arch = FWArch(BaseArch.FeedForwardNet1)
arch.SetParameter("ActivationFunction", ActivationFunction.ReLUFunction)
arch.SetParameter("NumberOfInputNodes", 28*28)
arch.SetParameter("NumberOfHiddenNodes", 1024)
arch.SetParameter("NumberOfOutputNodes", 10)
arch.Build()

def create_local_clients(train_ds_list: list, clients_list: list):
   for client_id in range(len(train_ds_list)):
      model = arch.CreateModel()
      client = Client(f"Client#{client_id}", IpAddr("127.0.0.1", 9911), TrainingHyperParameters(0.001, 0.9, 1e-7), train_ds_list[client_id], model, optim.SGD, nn.CrossEntropyLoss, "cpu")
      clients_list.append(client)
   
fedavg = FedAvg()



train_ds_list, test_ds = create_datasets(20, "MNIST", True, 0.4, 128, 256, True)

weights = [(float(len(dataloader.dataset)) / float(sum([len(dataloader.dataset) for  dataloader in train_ds_list]))) for dataloader in train_ds_list]
fedavg.args = weights

model = arch.CreateModel()
server = Server(IpAddr("127.0.0.1", 9911), fedavg, test_ds, model, optim.SGD, nn.CrossEntropyLoss, "cpu")
clients_list = []
create_local_clients(train_ds_list, clients_list)
server.start_training()
sg.theme('Dark Amber')

# STEP 1 define the layout
layout = [ 
            [sg.Text('Test')],
            [sg.Input()],
            [sg.Button('Button'), sg.Button('Exit')]
         ]

#STEP 2 - create the window
window = sg.Window('Federated Learning: A simple framework', layout)

# STEP3 - the event loop
while True:
    event, values = window.read()   # Read the event that happened and the values dictionary
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':     # If user closed window with X or if user clicked "Exit" button then exit
      break
    if event == 'Button':
      print('You pressed the button')
window.close()