include ResNet

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

