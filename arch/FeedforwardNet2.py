class FeedforwardNet2(nn.Module):
    def __init__(self):
        super(FeedforwardNet2, self).__init__()
        self.fc1 = nn.Linear(<<param1>>, <<param2>>)
        self.fc2 = nn.Linear(<<param2>>, <<param3>>)
        self.fc3 = nn.Linear(<<param3>>, <<param4>>)
        self.fc4 = nn.Linear(<<param4>>, <<param5>>)

    def forward(self, x):
        x = x.view(-1, <<param1>>)
        x = self.fc1(x)
        x = F.<<func1>>(x)
        x = self.fc2(x)
        x = F.<<func2>>(x)
        x = self.fc3(x)
        x = F.<<func3>>(x)
        x = self.fc4(x)
        x = F.<<func4>>(x)
        x = self.fc5(x)
        return x
