import torch
import torch.nn as nn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from torch.autograd import Variable
import torch.nn.functional as F
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pickle

num_input_channel = 1
batch_size = 1


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

        '''self.fc0 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.ReLU())'''

        self.fc1 = nn.Sequential(
            nn.Linear(5120, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(14336, 4096),
            nn.Dropout(0.5),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(11264, 4096),
            nn.Dropout(0.5),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        # final fc layer

        self.fc_final = nn.Sequential(
            nn.Linear(10240, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU())

    def forward(self, x):
        # x0 = x.view(-1, self.num_flat_features(x))

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

        # all concatenated features

        x1 = torch.cat((x11z, x21z), 1)
        x1 = torch.cat((x1, x31z), 1)

        x2 = torch.cat((x12z, x22z), 1)
        x2 = torch.cat((x2, x32z), 1)

        x3 = torch.cat((x13z, x23z), 1)
        x3 = torch.cat((x3, x33z), 1)

        # concatenated features fc layer

        # xz0 = self.fc0(x0)
        xz1 = self.fc1(x1)
        xz2 = self.fc2(x2)
        xz3 = self.fc3(x3)

        # all features concatenation

        # xz = torch.cat((xz0, xz1), 1)
        xz = torch.cat((xz1, xz2), 1)
        xz = torch.cat((xz, xz3), 1)

        # final fc layer

        out = self.fc_final(xz)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
print("Loading the dataset")
train_loader = pickle.load(open("sample_data/train_loader2.txt", 'rb'))
val_loader = pickle.load(open("sample_data/val_loader2.txt", 'rb'))
test_loader = pickle.load(open("sample_data/test_loader2.txt", 'rb'))

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))
print("Dataset is loaded")

resume_weights = "sample_data/checkpointBDS.pth.tar"
cuda = torch.cuda.is_available()

model = Column1()
if cuda:
    model.cuda()

checkpoint = torch.load(resume_weights, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])


def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list


def eval(model, test_loader):
    output_label = []
    output_target = []
    model.eval()
    for i, (data, target) in enumerate(test_loader):
        data, target = Variable(data, volatile=True), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)

        output = output.cpu()
        target = target.cpu()
        output = output.data.numpy()
        target = target.data.numpy()
        output = output.tolist()
        target = target.tolist()
        output = np.ravel(output)
        output_label.append(output)
        output_target.append(target)

    return output_label, output_target


print("Constructing dataset of SVM...")
x_train, y_train = eval(model, train_loader)
pickle.dump(x_train, open('sample_data/x_train.txt', 'wb'))
pickle.dump(y_train, open('sample_data/y_train.txt', 'wb'))

print("Dataset of SVM is constructed", end='\n\n\n')

print("SVM starts training...")
clf = svm.SVC(kernel='rbf')

save_model = "sample_data/svm_bn.sav"
x = np.array(pickle.load(open('sample_data/x_train.txt', 'rb')))
y = np.array(pickle.load(open('sample_data/y_train.txt', 'rb')))

print(np.shape(x))
print(np.shape(y))

X_train, X_validate, y_train, y_validate = train_test_split(x, y, test_size=0.2)
clf.fit(X_train, y_train)
training_score = clf.score(X_train, y_train)
validation_score = clf.score(X_validate, y_validate)
print("Training Accuracy : ", training_score * 100)
print("validation Accuracy : ", validation_score * 100)
pickle.dump(clf, open(save_model, 'wb'))

print("SVM completed it's training")

print("SVM starts testing...")
clf = pickle.load(open(save_model, 'rb'))
x_test, y_test = eval(model, test_loader)
y_test = np.reshape(y_test, (len(y_test), 1))
test_score = clf.score(x_test, y_test)
print("Testing Accuracy : ", test_score * 100)

y_pred = clf.predict(x_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
pickle.dump(cnf_matrix, open('sample_data/confusion_matrix_FDS.txt', 'wb'))
np.set_printoptions(precision=2)
print(cnf_matrix)

seaborn.heatmap(cnf_matrix, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

