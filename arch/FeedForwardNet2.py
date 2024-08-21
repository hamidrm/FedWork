class FeedForwardNet2(nn.Module):
    def __init__(self):
        super(FeedForwardNet2, self).__init__()
        self.fc1 = nn.Linear(<<NumberOfInputNodes:integer>>, <<NumberOfHiddenNodes1:integer>>)
        self.fc2 = nn.Linear(<<NumberOfHiddenNodes1>>, <<NumberOfHiddenNodes2:integer>>)
        self.fc3 = nn.Linear(<<NumberOfHiddenNodes2>>, <<NumberOfHiddenNodes3:integer>>)
        self.fc4 = nn.Linear(<<NumberOfHiddenNodes3>>, <<NumberOfOutputNodes:integer>>)

    def forward(self, x):
        x = x.view(-1, <<NumberOfInputNodes>>)
        x = self.fc1(x)
        x = F.<<ActivationFunction:act_fn>>(x)
        x = self.fc2(x)
        x = F.<<ActivationFunction:act_fn>>(x)
        x = self.fc3(x)
        x = F.<<ActivationFunction:act_fn>>(x)
        x = self.fc4(x)
        return x
