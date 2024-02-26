class FeedforwardNet1(nn.Module):
    def __init__(self):
        super(FeedforwardNet1, self).__init__()
        self.fc1 = nn.Linear(<<param1>>, <<param2>>)
        self.fc2 = nn.Linear(<<param2>>, <<param3>>)

    def forward(self, x):
        x = x.view(-1, <<param1>>)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
