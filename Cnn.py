import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)  # 输入通道3（彩色图像），输出通道6
        self.b1 = nn.BatchNorm2d(6)
        self.a1 = nn.ReLU()
        self.p1 = nn.MaxPool2d(2, 2)
        self.d1 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(6 * 16 * 16, 128)
        self.d2 = nn.Dropout(0.2)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y

# 实例化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Baseline().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    model.eval()
    running_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct_test / total_test

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {100 * train_accuracy:.2f}%, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {100 * test_accuracy:.2f}%')

    # 记录历史数据
    train_loss_history.append(train_loss)
    train_acc_history.append(train_accuracy)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_accuracy)

# 保存模型
torch.save(model.state_dict(), 'baseline_model.pth')

# 绘制损失和准确率曲线
epochs_range = range(1, num_epochs + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss_history, label='Train Loss')
plt.plot(epochs_range, test_loss_history, label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc_history, label='Train Accuracy')
plt.plot(epochs_range, test_acc_history, label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()