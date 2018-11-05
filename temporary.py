import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision.datasets as dsets
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import time
import os.path
import numpy as np
import pickle

num_epochs = 200
batch_size = 500
learning_rate = 0.001
print_every = 1
best_accuracy = torch.FloatTensor([0])
start_epoch = 0
num_input_channel = 1

resume_weights = "sample_data/checkpointBDSM.pth.tar"

cuda = torch.cuda.is_available()

torch.manual_seed(1)

if cuda:
    torch.cuda.manual_seed(1)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    torchvision.transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

print("Loading the dataset")
# train_set = torchvision.datasets.ImageFolder(root="BanglaDigit/Train", transform=transform)
train_set = dsets.MNIST(root='/input', train=True, download=True, transform=transform)
indices = list(range(len(train_set)))
val_split = 10000

val_idx = np.random.choice(indices, size=val_split, replace=False)
train_idx = list(set(indices) - set(val_idx))

val_sampler = SubsetRandomSampler(val_idx)
val_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, sampler=val_sampler, shuffle=False)
val_loader2 = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, sampler=val_sampler, shuffle=False)

train_sampler = SubsetRandomSampler(train_idx)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler,
                                           shuffle=False)
train_loader2 = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, sampler=train_sampler, shuffle=False)

# test_set = torchvision.datasets.ImageFolder(root="BanglaDigit/Test", transform=transform)
test_set = dsets.MNIST(root='/input', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
test_loader2 = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
print("Dataset is loaded")

print("Saving the dataset...")
pickle.dump(train_loader, open("sample_data/train_loader.txt", 'wb'))
pickle.dump(val_loader, open("sample_data/val_loader.txt", 'wb'))
pickle.dump(test_loader, open("sample_data/test_loader.txt", 'wb'))

pickle.dump(train_loader2, open("sample_data/train_loader2.txt", 'wb'))
pickle.dump(val_loader2, open("sample_data/val_loader2.txt", 'wb'))
pickle.dump(test_loader2, open("sample_data/test_loader2.txt", 'wb'))

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))
print("Dataset is saved")


def train(model, optimizer, train_loader, loss_fun):
    average_time = 0
    total = 0
    acc = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        batch_time = time.time()
        images = Variable(images)
        labels = Variable(labels)

        if cuda:
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fun(outputs, labels)

        if cuda:
            loss.cpu()

        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_time
        average_time += batch_time

        total += labels.size(0)
        prediction = outputs.data.max(1)[1]
        correct = prediction.eq(labels.data).sum()
        acc += correct

        if (i + 1) % print_every == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Accuracy: %.4f, Batch time: %f'
                  % (epoch + 1,
                     num_epochs,
                     i + 1,
                     len(train_loader),
                     loss.data[0],
                     acc / total,
                     average_time / print_every))


def eval(model, test_loader):
    model.eval()

    acc = 0
    total = 0
    for i, (data, labels) in enumerate(test_loader):
        data, labels = Variable(data), Variable(labels)
        if cuda:
            data, labels = data.cuda(), labels.cuda()

        data = data.squeeze(0)
        labels = labels.squeeze(0)

        outputs = model(data)
        if cuda:
            outputs.cpu()

        total += labels.size(0)
        prediction = outputs.data.max(1)[1]
        correct = prediction.eq(labels.data).sum()
        acc += correct
    return acc / total


def save_checkpoint(state, is_best, filename="sample_data/checkpointBDS1.pth.tar"):
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


class Column1(nn.Module):
    def __init__(self):
        super(Column1, self).__init__()

        self.layer11 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=3, stride=2, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer12 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer13 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=7, stride=1, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer21 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=5, stride=1, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer23 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer31 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=7, stride=1, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer32 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer33 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # first column fc layer

        self.fc11 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.ReLU())

        self.fc12 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc13 = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.ReLU())

        # second column fc layer

        self.fc21 = nn.Sequential(
            nn.Linear(7200, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc22 = nn.Sequential(
            nn.Linear(10816, 4096),
            nn.Dropout(0.5),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        self.fc23 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        # third column fc layer

        self.fc31 = nn.Sequential(
            nn.Linear(6272, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc32 = nn.Sequential(
            nn.Linear(16384, 8192),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8192),
            nn.ReLU())

        self.fc33 = nn.Sequential(
            nn.Linear(16384, 8192),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8192),
            nn.ReLU())

        # concatenated features fc layer

        self.fc0 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.ReLU())

        # row-wise skip-concatenation fc layer

        self.fc1r = nn.Sequential(
            nn.Linear(5120, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc2r = nn.Sequential(
            nn.Linear(14336, 4096),
            nn.Dropout(0.5),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        self.fc3r = nn.Sequential(
            nn.Linear(11264, 4096),
            nn.Dropout(0.5),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        # column-wise skip-concatenation fc layer

        self.fc1c = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc2c = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.Dropout(0.5),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        self.fc3c = nn.Sequential(
            nn.Linear(18432, 8192),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8192),
            nn.ReLU())

        # final row-wise fc layer

        self.fc_r_final = nn.Sequential(
            nn.Linear(10752, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        # final column-wise fc layer

        self.fc_c_final = nn.Sequential(
            nn.Linear(14848, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        # final fc layer

        self.fc_final = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU())

    def forward(self, x):
        x0 = x.view(-1, self.num_flat_features(x))

        # for first column
        x3 = self.layer11(x)
        x11 = x3.view(-1, self.num_flat_features(x3))

        x3 = self.layer12(x3)
        x12 = x3.view(-1, self.num_flat_features(x3))

        x3 = self.layer13(x3)
        x13 = x3.view(-1, self.num_flat_features(x3))

        # for second column
        x5 = self.layer21(x)
        x21 = x5.view(-1, self.num_flat_features(x5))

        x5 = self.layer22(x5)
        x22 = x5.view(-1, self.num_flat_features(x5))

        x5 = self.layer23(x5)
        x23 = x5.view(-1, self.num_flat_features(x5))

        # for third column
        x7 = self.layer31(x)
        x31 = x7.view(-1, self.num_flat_features(x7))

        x7 = self.layer32(x7)
        x32 = x7.view(-1, self.num_flat_features(x7))

        x7 = self.layer33(x7)
        x33 = x7.view(-1, self.num_flat_features(x7))

        # features from first column

        x11z = self.fc11(x11)
        x12z = self.fc12(x12)
        x13z = self.fc13(x13)

        # features from second column

        x21z = self.fc21(x21)
        x22z = self.fc22(x22)
        x23z = self.fc23(x23)

        # features from third column

        x31z = self.fc31(x31)
        x32z = self.fc32(x32)
        x33z = self.fc33(x33)

        # all concatenated features row-wise

        x1r = torch.cat((x11z, x21z), 1)
        x1r = torch.cat((x1r, x31z), 1)

        x2r = torch.cat((x12z, x22z), 1)
        x2r = torch.cat((x2r, x32z), 1)

        x3r = torch.cat((x13z, x23z), 1)
        x3r = torch.cat((x3r, x33z), 1)

        # all concatenated features column-wise

        x1c = torch.cat((x11z, x12z), 1)
        x1c = torch.cat((x1c, x13z), 1)

        x2c = torch.cat((x21z, x22z), 1)
        x2c = torch.cat((x2c, x23z), 1)

        x3c = torch.cat((x31z, x32z), 1)
        x3c = torch.cat((x3c, x33z), 1)

        # concatenated features fc layer

        xz0 = self.fc0(x0)

        # row-wise
        xz1r = self.fc1r(x1r)
        xz2r = self.fc2r(x2r)
        xz3r = self.fc3r(x3r)

        # column-wise

        xz1c = self.fc1c(x1c)
        xz2c = self.fc2c(x2c)
        xz3c = self.fc3c(x3c)

        # all features concatenation

        xzr = torch.cat((xz0, xz1r), 1)
        xzr = torch.cat((xzr, xz2r), 1)
        xzr = torch.cat((xzr, xz3r), 1)

        # final row-wise fc layer

        outr = self.fc_r_final(xzr)

        xzc = torch.cat((xz0, xz1c), 1)
        xzc = torch.cat((xzc, xz2c), 1)
        xzc = torch.cat((xzc, xz3c), 1)

        # final column-wise fc layer

        outc = self.fc_c_final(xzc)

        out = torch.cat((outr, outc), 1)

        outz = self.fc_final(out)

        return outz

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = Column1()
if cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
if cuda:
    criterion.cuda()

total_step = len(train_loader)

if os.path.isfile(resume_weights):
    print("=> loading checkpoint '{}' ...".format(resume_weights))
    if cuda:
        checkpoint = torch.load(resume_weights)
    else:
        checkpoint = torch.load(resume_weights, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))

for epoch in range(num_epochs):
    print(learning_rate)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    if learning_rate >= 0.0003:
        learning_rate = learning_rate * 0.993

    train(model, optimizer, train_loader, criterion)
    acc = eval(model, val_loader)
    print('=> Validation set: Accuracy: {:.2f}%'.format(acc * 100))
    acc = torch.FloatTensor([acc])

    is_best = bool(acc.numpy() > best_accuracy.numpy())

    best_accuracy = torch.FloatTensor(max(acc.numpy(), best_accuracy.numpy()))

    save_checkpoint({
        'epoch': start_epoch + epoch + 1,
        'state_dict': model.state_dict(),
        'best_accuracy': best_accuracy
    }, is_best)

    test_acc = eval(model, test_loader)
    print('=> Test set: Accuracy: {:.2f}%'.format(test_acc * 100))

test_acc = eval(model, test_loader)
print('=> Test set: Accuracy: {:.2f}%'.format(test_acc * 100))

