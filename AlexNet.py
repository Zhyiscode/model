import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 加载CIFAR-10数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
print('加载完成')

# 定义ConvBNRelu模型类
class ConvBNRelu(nn.Module):
    def __init__(self):
        super(ConvBNRelu, self).__init__()
        # 定义模型层
        self.c1 = nn.Conv2d(3, 96, kernel_size=3)
        self.b1 = nn.BatchNorm2d(96)
        self.a1 = nn.ReLU()
        self.p1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.c2 = nn.Conv2d(96, 256, kernel_size=3)
        self.b2 = nn.BatchNorm2d(256)
        self.a2 = nn.ReLU()
        self.p2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.c3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.c5 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.p3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(384 * 2 * 2, 2048)
        self.d1 = nn.Dropout(0.5)
        self.f2 = nn.Linear(2048, 2048)
        self.d2 = nn.Dropout(0.5)
        self.f3 = nn.Linear(2048, 10)

    def forward(self, x):
        # 定义模型的前向传播
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        print('完成一次前向')
        return y


# 创建模型实例
model = ConvBNRelu()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型，并可视化损失率和准确率
num_epochs = 10
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # 训练模式
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        print('循环一次')

    average_train_loss = running_loss / len(trainloader)
    train_losses.append(average_train_loss)
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)

    # 测试模式
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    average_test_loss = test_loss / len(testloader)
    test_losses.append(average_test_loss)
    test_accuracy = correct_test / total_test
    test_accuracies.append(test_accuracy)

    # 打印和可视化
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {average_train_loss:.4f}, Train Acc: {train_accuracy:.2%}, '
          f'Test Loss: {average_test_loss:.4f}, Test Acc: {test_accuracy:.2%}')

# 可视化损失率和准确率
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
