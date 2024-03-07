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


arch = FWArch(BaseArch.FeedForwardNet1)
arch.SetParameter("NumberOfInputNodes", 28 * 28)
arch.SetParameter("NumberOfHiddenNodes", 128)
arch.SetParameter("NumberOfOutputNodes", 10)
arch.SetParameter("ActivationFunction", ActivationFunction.ReLUFunction)
arch.Build()


def client_evt(name, evt, data):
   if evt == COMM_EVT_TRAINING_START:
      print(f'Client Event, Client: "{name}", Evt: "{evt}", Data:\n "{data}"')

def server_evt(evt, client, data):
   print(f'Server Event, Client: "{client.name}", Evt: "{evt}", Data:\n "{data}"')

def create_local_clients(train_ds_list: list, clients_list: list):
   for client_id in range(len(train_ds_list)):
      model = arch.CreateModel()
      client = Client(f"Client#{client_id}", IpAddr("127.0.0.1", 9911), TrainingHyperParameters(0.001, 0.9, 1e-7), train_ds_list[client_id], model, optim.SGD, nn.CrossEntropyLoss, "cpu")
      #client.send_notification_to_server(COMM_EVT_TRAINING_START, 5, 0)
      clients_list.append(client)
   



#server = ServerComm("127.0.0.1", 62120, server_evt)

#client1 = ClientComm("Client#1", "127.0.0.1", 62120, client_evt)
#client2 = ClientComm("Client#2", "127.0.0.1", 62120, client_evt)

#server.send_command("Client#1", COMM_HEADER_CMD_START_TRAINNING, 0)
fedavg = FedAvg()
train_ds_list, test_ds = create_datasets(4, "MNIST", True, 0.5, 128, 256, True)
model = arch.CreateModel()
server = Server(IpAddr("127.0.0.1", 9911), fedavg, test_ds, 3, model, optim.SGD, nn.CrossEntropyLoss, "cpu")
clients_list = []
create_local_clients(train_ds_list, clients_list)
server.start_round(5, [100, 200], 0.01)
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