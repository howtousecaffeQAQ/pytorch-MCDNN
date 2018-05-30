#coding=utf-8
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import os
import torchvision.transforms as transforms
import argparse
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(360, 150)
        self.fc2 = nn.Linear(150, 10)
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3))
        x = x.view(-1, 360)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def read_traindata(path, transform, width):
    trainstr = 'train' + str(width)
    traindir = os.path.join(path, trainstr)
    train = datasets.ImageFolder(traindir, transform)
    train_loader = torch.utils.data.DataLoader(train, 256, shuffle=True, num_workers=2, pin_memory=True)
    return train_loader
def read_testdata(path, transform):
    testdir = os.path.join(path, 'test')
    test = datasets.ImageFolder(testdir, transform)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1000, shuffle=True, num_workers=2, pin_memory=True )
    return test_loader

def single_imageprocess(batch_data, transform):
    length = len(batch_data)
    for i in range(0, length):
        batch_data[i] = transform(batch_data[i])
    return batch_data

def train(args, model, device, train_loader, optimizer, epoch, transform):
    model.train().cuda(device)
    k = enumerate(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = single_imageprocess(data, transform)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

def test_singlemodel(args,model,device,test_loader):
    model.eval().cuda(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_allmodel(args, modellist, device, test_loader):
    length = len(modellist)
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        test_loss = 0
        with torch.no_grad():
            for i in range(0, length):
                modellist[i].eval()
                if i == 0:
                    output = modellist[i](data)
                else:
                    output += modellist[i](data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nall models Test set: , Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load local dataset')
    parser.add_argument('--path', metavar='PATH', required=True, help='path to dataset')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()
    transform = torchvision.transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((28, 28))
                                       , transforms.ToTensor()])
    transform_single = torchvision.transforms.Compose([transforms.ToPILImage(),transforms.RandomAffine(degrees = 15,translate = (0.15, 0.15),scale = (0.85, 1.15)),
                                                transforms.Resize(28),transforms.ToTensor()])
    # ,  ,
    loaderlist = []
    dataset_widthlist = [0,12,16,20,24,26,32]
    for i in dataset_widthlist:
        train_loader = read_traindata(args.path, transform, i)
        loaderlist.append(train_loader)
    test_loader = read_testdata(args.path, transform)
    torch.manual_seed(1)
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    modellist = []
    for width_loadernum in range(0,len(loaderlist)):
        for modelnum in range(0,5):
            model = Net().cuda()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            for epoch in range(1, args.epochs + 100):
                train(args, model, device, loaderlist[width_loadernum], optimizer, epoch, transform_single)
                test_singlemodel(args, model, device, test_loader)
            modelstr = 'model'+'width_'+ str(dataset_widthlist[width_loadernum])+'m_'+str(modelnum) + '.pkl'
            torch.save(model, modelstr)
            modellist.append(model)
            test_allmodel(args,modellist,device,test_loader)
        modellist.append(model)
        print('one dataset with different width:')
        test_allmodel(args, modellist, device, test_loader)