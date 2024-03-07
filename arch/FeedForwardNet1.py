class FeedForwardNet1(nn.Module):
    def __init__(self):
        super(FeedForwardNet1, self).__init__()
        self.fc1 = nn.Linear(<<NumberOfInputNodes:integer>>, <<NumberOfHiddenNodes:integer>>)
        self.fc2 = nn.Linear(<<NumberOfHiddenNodes>>, <<NumberOfOutputNodes:integer>>)

    def forward(self, x):
        x = x.view(-1, <<NumberOfInputNodes>>)
        x = self.fc1(x)
        x = F.<<ActivationFunction:act_fn>>(x)
        x = self.fc2(x)
        return x
