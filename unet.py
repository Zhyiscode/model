'''手写体数字的识别'''

import tensorflow as tf  # 导入 TensorFlow 库
from tensorflow.keras import layers, models  # 从 Keras 中导入特定组件
from tensorflow.keras.datasets import mnist  # 导入 MNIST 数据集
import matplotlib.pyplot as plt  # 导入 matplotlib 进行可视化

# 加载 MNIST 数据集，分为训练集和测试集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理：重新塑形图像并将像素值归一化
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 将标签转换为分类的 one-hot 编码
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# 构建神经网络模型
model = models.Sequential()
# 顺序模型用于堆叠层
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 添加一个 2D 卷积层，有 32 个过滤器，每个大小为 3x3，使用 ReLU 激活函数，输入形状为 (28, 28, 1)
model.add(layers.MaxPooling2D((2, 2)))
# 添加一个最大池化层（2x2 窗口）
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 添加另一个具有 64 个过滤器的卷积层
model.add(layers.MaxPooling2D((2, 2)))
# 再添加一个最大池化层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 再添加一个具有 64 个过滤器的卷积层
model.add(layers.Flatten())
# 展平层，将多维输入展平为一维
model.add(layers.Dense(64, activation='relu'))
# 全连接层，具有 64 个神经元，使用 ReLU 激活函数
model.add(layers.Dense(10, activation='softmax'))
# 输出层，具有 10 个神经元，使用 softmax 激活函数

# 编译模型
model.compile(optimizer='adam',  # 优化器为 Adam
              loss='categorical_crossentropy',  # 损失函数为分类交叉熵
              metrics=['accuracy'])  # 评估指标为准确率

# 训练模型
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))
# 使用训练集训练模型，共进行 5 轮训练，每批次包含 64 个样本，同时验证测试集

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')  # 打印测试集的准确率

# 绘制训练过程中的准确率和损失曲线
# 绘制训练准确率曲线
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')  # 绘制验证准确率曲线
plt.xlabel('Epoch')  # 设置 x 轴标签为 Epoch
plt.ylabel('Accuracy')  # 设置 y 轴标签为 Accuracy
plt.legend(loc='lower right')  # 显示图例，位置为右下角
plt.show()  # 显示绘制的图形


