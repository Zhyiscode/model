import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import os
from skimage.color import rgb2gray

# 加载数据
def load_data(data_dir):
    images = []
    labels = []

    for filename in os.listdir( data_dir ):
        if "image" in filename:
            # 加载图像
            image_path = os.path.join( data_dir, filename )
            image = io.imread( image_path )

            # 加载相应的分割标签数据
            mask_filename = filename.replace( "image", "mask" )
            mask_path = os.path.join( data_dir, mask_filename )
            mask = io.imread( mask_path )

            # 确保图像和标签形状一致
            if image.shape[:2] == mask.shape[:2]:
                images.append( image )
                labels.append( rgb2gray( mask ) )  # 如果标签是彩色的，可以转换为灰度图像

    return np.array( images ), np.array( labels )


# 指定包含图像和标签的数据目录
data_directory = "D:\pycharmyunxing\shuju"
images, labels = load_data( data_directory )


# 数据预处理
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(images, labels, target_size=(256, 256)):
    # 调整图像大小
    images_resized = [resize(image, target_size, mode='reflect', anti_aliasing=True) for image in images]

    # 归一化图像（可选）
    images_normalized = [(image - np.min(image)) / (np.max(image) - np.min(image)) for image in images_resized]

    # 归一化标签（可选）
    scaler = MinMaxScaler()
    labels_flat = [label.flatten() for label in labels]
    labels_normalized = [scaler.fit_transform(label.reshape(-1, 1)).reshape(label.shape) for label in labels_flat]

    return np.array(images_normalized), np.array(labels_normalized)

# 调用数据预处理函数
target_size = (256, 256)
images_preprocessed, labels_preprocessed = preprocess_data(images, labels, target_size=target_size)



# 定义U-Net模型
def unet_model(input_shape=(256, 256, 1)):
    inputs = Input( input_shape )

    # 编码器
    conv1 = Conv2D( 64, 3, activation='relu', padding='same' )( inputs )
    conv1 = Conv2D( 64, 3, activation='relu', padding='same' )( conv1 )
    pool1 = MaxPooling2D( pool_size=(2, 2) )( conv1 )

    conv2 = Conv2D( 128, 3, activation='relu', padding='same' )( pool1 )
    conv2 = Conv2D( 128, 3, activation='relu', padding='same' )( conv2 )
    pool2 = MaxPooling2D( pool_size=(2, 2) )( conv2 )

    conv3 = Conv2D( 256, 3, activation='relu', padding='same' )( pool2 )
    conv3 = Conv2D( 256, 3, activation='relu', padding='same' )( conv3 )
    pool3 = MaxPooling2D( pool_size=(2, 2) )( conv3 )

    # 中间层
    conv4 = Conv2D( 512, 3, activation='relu', padding='same' )( pool3 )
    conv4 = Conv2D( 512, 3, activation='relu', padding='same' )( conv4 )

    # 解码器
    up5 = concatenate( [UpSampling2D( size=(2, 2) )( conv4 ), conv3], axis=-1 )
    conv5 = Conv2D( 256, 3, activation='relu', padding='same' )( up5 )
    conv5 = Conv2D( 256, 3, activation='relu', padding='same' )( conv5 )

    up6 = concatenate( [UpSampling2D( size=(2, 2) )( conv5 ), conv2], axis=-1 )
    conv6 = Conv2D( 128, 3, activation='relu', padding='same' )( up6 )
    conv6 = Conv2D( 128, 3, activation='relu', padding='same' )( conv6 )

    up7 = concatenate( [UpSampling2D( size=(2, 2) )( conv6 ), conv1], axis=-1 )
    conv7 = Conv2D( 64, 3, activation='relu', padding='same' )( up7 )
    conv7 = Conv2D( 64, 3, activation='relu', padding='same' )( conv7 )

    outputs = Conv2D( 1, 1, activation='sigmoid' )( conv7 )

    model = Model( inputs=inputs, outputs=outputs )
    return model


# 加载和预处理数据
images, labels = load_data()
images, labels = preprocess_data( images, labels )

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split( images, labels, test_size=0.2, random_state=42 )

# 构建和编译模型
model = unet_model( input_shape=(256, 256, 1) )
model.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )

# 训练模型
model.fit( X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test) )

# 评估模型
loss, accuracy = model.evaluate( X_test, y_test )
print( f"Test Loss: {loss}, Test Accuracy: {accuracy}" )

# TODO: 添加你的测试图像路径
test_image_path = "/path/to/your/test/image.png"

# 读取测试图像
test_image = io.imread(test_image_path)

# 预处理测试图像（与训练数据相同的预处理步骤）
test_image_resized = resize(test_image, target_size, mode='reflect', anti_aliasing=True)
test_image_normalized = (test_image_resized - np.min(test_image_resized)) / (np.max(test_image_resized) - np.min(test_image_resized))

# 将图像添加一个批次维度
test_image_input = np.expand_dims(test_image_normalized, axis=0)

# 使用训练好的模型进行预测
predicted_mask = model.predict(test_image_input)

# 后处理：二值化预测结果（根据任务需要）
threshold = 0.5  # 调整阈值根据实际情况
binary_mask = (predicted_mask > threshold).astype(np.uint8)

# 显示原始图像和预测结果
plt.subplot(1, 2, 1)
plt.imshow(test_image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(binary_mask[0, ..., 0], cmap='gray')
plt.title('Predicted Mask')

plt.show()


# 绘制预测结果
# TODO: 添加绘制预测结果的代码

