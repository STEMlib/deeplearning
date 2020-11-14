import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import sys

# choose which data set to use {MNIST (1), CIFAR10 (0)}

def import_data(whichset = 1):
# import data

    if whichset == 1:
        transform_train = transforms.Compose([transforms.ToTensor()])
        img_shape = 28*28
        train = torchvision.datasets.MNIST('./data',train=True,download=False,transform = transform_train)
        test = torchvision.datasets.MNIST('./data',train=False,download=False,transform = transform_train)
    elif whichset == 0:
        transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        img_shape = 3*32*32
        train = torchvision.datasets.CIFAR10('./data',train=True,download=False,transform = transform_train)
        test = torchvision.datasets.CIFAR10('./data',train=False,download=False,transform = transform_train)

    trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
    testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)
    
    return img_shape, trainset, testset
    
class logreg_net(nn.Module):
    '''
    Logistic Regression using a neural net is:
        - One input layer
        - One hidden layer
        - One output layer
    '''
    # image shape
    #img_shape = 3*32*32
    #img_shape = 28*28

    #initialize
    def __init__(self,img_shape):    
        super().__init__()
        self.layer1 = nn.Linear(img_shape, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 10)


    # forward
    def forward(self,x):        
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return F.log_softmax(x,dim=1)
    
    
def main(whichset = 1):
    img_shape, trainset, testset = import_data(whichset)
    model = logreg_net(img_shape)
    
    # train
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    EPOCHS = 5
    for epoch in range(EPOCHS):
        for data in trainset:
            X, y = data
            model.zero_grad()
            output = model(X.view(-1,img_shape))
            loss = F.nll_loss(output,y)
            loss.backward()
            optimizer.step()
        print("Loss = ", loss)
    
    # Evaluate
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testset:
            X, y = data
            output = model(X.view(-1,img_shape))
            #print(output)
            for idx, i in enumerate(output):
                #print(torch.argmax(i), y[idx])
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))


if __name__ == '__main__':
    main()