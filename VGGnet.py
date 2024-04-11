import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 加载CIFAR-10数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# 转换数据为PyTorch张量
x_train = torch.Tensor(train_dataset.data.transpose((0, 3, 1, 2))) / 255.0
y_train = torch.LongTensor(train_dataset.targets)

x_test = torch.Tensor(test_dataset.data.transpose((0, 3, 1, 2))) / 255.0
y_test = torch.LongTensor(test_dataset.targets)
print('加载完成')

# 定义模型
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d( 3, 64, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 64 ),
            nn.ReLU(),

            nn.Conv2d( 64, 64, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 64 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ),
            nn.Dropout( 0.2 ),

            nn.Conv2d( 64, 128, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 128 ),
            nn.ReLU(),

            nn.Conv2d( 128, 128, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 128 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ),
            nn.Dropout( 0.2 ),

            nn.Conv2d( 128, 256, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 256 ),
            nn.ReLU(),

            nn.Conv2d( 256, 256, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 256 ),
            nn.ReLU(),

            nn.Conv2d( 256, 256, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 256 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ),
            nn.Dropout( 0.2 ),

            nn.Conv2d( 256, 512, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 512 ),
            nn.ReLU(),

            nn.Conv2d( 512, 512, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 512 ),
            nn.ReLU(),

            nn.Conv2d( 512, 512, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 512 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ),
            nn.Dropout( 0.2 ),

            nn.Conv2d( 512, 512, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 512 ),
            nn.ReLU(),

            nn.Conv2d( 512, 512, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 512 ),
            nn.ReLU(),

            nn.Conv2d( 512, 512, kernel_size=3, padding=1 ),
            nn.BatchNorm2d( 512 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ),
            nn.Dropout( 0.2 ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear( 512, 512 ),
            nn.ReLU(),
            nn.Dropout( 0.2 ),

            nn.Linear( 512, 512 ),
            nn.ReLU(),
            nn.Dropout( 0.2 ),

            nn.Linear( 512, 10 ),
            nn.Softmax( dim=1 ),
        )

    def forward(self, x):
        x = self.features( x )
        x = self.classifier( x )
        print('完成一次前向')
        return x

model = Baseline()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 转换为PyTorch的数据加载器
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32, shuffle=False)

# 训练模型并可视化
epochs = 10
losses = []
accuracies = []

for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 在验证集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    accuracies.append(accuracy)
    losses.append(loss.item())

    print(f'Epoch {epoch + 1}/{epochs}, Validation Accuracy: {accuracy}, Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), "./baseline.pth")

# 可视化损失和准确率
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()