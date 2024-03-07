class FeedforwardNet2(nn.Module):
    def __init__(self):
        super(FeedforwardNet2, self).__init__()
        self.fc1 = nn.Linear(<<param1:integer:NumberOfInputsNodes>>, <<param2:integer:NumberOfHidden1Nodes>>)
        self.fc2 = nn.Linear(<<param2:integer:NumberOfHidden1Nodes>>, <<param3:integer:NumberOfHidden2Nodes>>)
        self.fc3 = nn.Linear(<<param3:integer:NumberOfHidden2Nodes>>, <<param4:integer:NumberOfHidden3Nodes>>)
        self.fc4 = nn.Linear(<<param4:integer:NumberOfHidden3Nodes>>, <<param5:integer:NumberOfOutputNodes>>)

    def forward(self, x):
        x = x.view(-1, <<param1:integer>>)
        x = self.fc1(x)
        x = F.<<func1:act_fn:InputLayerActivation>>(x)
        x = self.fc2(x)
        x = F.<<func2:act_fn:InputLayerActivation>>(x)
        x = self.fc3(x)
        x = F.<<func3:act_fn:InputLayerActivation>>(x)
        x = self.fc4(x)
        x = F.<<func4:act_fn:InputLayerActivation>>(x)
        x = self.fc5(x)
        return x
