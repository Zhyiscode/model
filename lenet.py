import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt


# 定义 PyTorch 中的 LeNet 模型
class LeNet( nn.Module ):
    def __init__(self):
        super( LeNet, self ).__init__()
        self.c1 = nn.Conv2d( 3, 6, kernel_size=(5, 5), stride=1, padding=0 )
        self.p1 = nn.MaxPool2d( kernel_size=(2, 2), stride=2 )
        self.c2 = nn.Conv2d( 6, 16, kernel_size=(5, 5), stride=1, padding=0 )
        self.p2 = nn.MaxPool2d( kernel_size=(2, 2), stride=2 )
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear( 16 * 5 * 5, 120 )
        self.f2 = nn.Linear( 120, 84 )
        self.f3 = nn.Linear( 84, 10 )

    def forward(self, x):
        x = torch.sigmoid( self.c1( x ) )
        x = self.p1( x )
        x = torch.sigmoid( self.c2( x ) )
        x = self.p2( x )
        x = self.flatten( x )
        x = torch.sigmoid( self.f1( x ) )
        x = torch.sigmoid( self.f2( x ) )
        y = torch.softmax( self.f3( x ), dim=1 )
        print('完成一次前向')
        return y


# 使用 torchvision 加载 CIFAR-10 数据集
transform = transforms.Compose( [transforms.ToTensor()] )
train_dataset = datasets.CIFAR10( root='./data', train=True, transform=transform, download=True )
test_dataset = datasets.CIFAR10( root='./data', train=False, transform=transform, download=True )

train_loader = torch.utils.data.DataLoader( dataset=train_dataset, batch_size=32, shuffle=True )
test_loader = torch.utils.data.DataLoader( dataset=test_dataset, batch_size=32, shuffle=False )

# 初始化模型、损失函数和优化器
model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam( model.parameters() )

# 训练循环
num_epochs = 10
train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

for epoch in range( num_epochs ):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0

    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model( batch_data )
        loss = criterion( outputs, batch_labels )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max( outputs.data, 1 )
        total_train += batch_labels.size( 0 )
        correct_train += (predicted == batch_labels).sum().item()
        print('循环一次')

    train_loss = running_loss / len( train_loader )
    train_accuracy = correct_train / total_train
    train_loss_list.append( train_loss )
    train_acc_list.append( train_accuracy )

    # Validation
    model.eval()
    running_val_loss, correct_val, total_val = 0.0, 0, 0

    with torch.no_grad():
        for val_data, val_labels in test_loader:
            val_outputs = model( val_data )
            val_loss = criterion( val_outputs, val_labels )
            running_val_loss += val_loss.item()
            _, predicted_val = torch.max( val_outputs.data, 1 )
            total_val += val_labels.size( 0 )
            correct_val += (predicted_val == val_labels).sum().item()

    val_loss = running_val_loss / len( test_loader )
    val_accuracy = correct_val / total_val
    val_loss_list.append( val_loss )
    val_acc_list.append( val_accuracy )

    print( f'Epoch [{epoch + 1}/{num_epochs}], '
           f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, '
           f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}' )

# 保存 PyTorch 模型
torch.save( model.state_dict(), './LeNet.pth' )

# 可视化训练历史
plt.figure( figsize=(10, 4) )
plt.subplot( 1, 2, 1 )
plt.plot( train_loss_list, label='Training Loss' )
plt.plot( val_loss_list, label='Validation Loss' )
plt.title( 'Training and Validation Loss' )
plt.xlabel( 'Epoch' )
plt.ylabel( 'Loss' )
plt.legend()

plt.subplot( 1, 2, 2 )
plt.plot( train_acc_list, label='Training Accuracy' )
plt.plot( val_acc_list, label='Validation Accuracy' )
plt.title( 'Training and Validation Accuracy' )
plt.xlabel( 'Epoch' )
plt.ylabel( 'Accuracy' )
plt.legend()

plt.tight_layout()
plt.show()
