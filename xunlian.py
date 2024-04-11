import os
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skimage import io, transform
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import matplotlib.pyplot as plt
import cv2
from PIL import Image


# ...（用于加载和预处理数据的先前代码）
# 加载数据
def load_data(data_dir):
    # 构建两个列表，一个是image表示存储图像的列表，一个是labels表示存储标签的列表，都是空列表。
    images = []
    labels = []
# 遍历所填文件目录data_dir 所有的文件名称，赋给filename
    for filename in os.listdir( data_dir ):
        # if判断语句，筛选出文件名称中包含字符串“image”的文件
        if "image" in filename:
            # 加载图像os.path.join完善所遍历文件名称带有“image”的图像文件路径
            # image_path变量所表示的就是图像的路径。
            image_path = os.path.join( data_dir, filename )
            # io.imread函数用于读取加载图像，（）内填image_path,表示图像路径。
            image = io.imread( image_path )

            # 加载相应的分割标签数据
            # 构建相应的掩模图像文件名，.replace函数用于将字符串 filename 中所有出现的子字符串 "image" 替换为 "mask"。
            '''mask_filename = filename.replace( "image", "mask" )'''

            # 完善掩模图像的文件路径
            mask_path = os.path.join( data_dir, filename )
            # imread函数读取加载掩模图像。
            # 将 RGB 图像转换为灰度图像
            mask = io.imread( mask_path )
            mask = cv2.cvtColor( mask, cv2.COLOR_RGB2GRAY )

            # 确保图像和标签形状一致
            # if判断图像形状和掩模图像形状是否相同
            # .shape[:2]，.shape函数是显示图像的参数，如长度，宽度和通道数，这里使用了一个切片操作[:2]只提取图像中的前两个特征即长宽。
            print( f"Image path: {image_path}, Image shape: {image.shape}" )
            print( f"Mask path: {mask_path}, Mask shape: {mask.shape}" )

            if image.shape[:2] == mask.shape[:2]:
                # 如果形状相同即将相应的image和mask填入最开始定义的images和labels两个列表中。
                images.append( image )
                labels.append( mask )
# np.array 函数将 images 和 labels 转换为 NumPy 数组，并将它们作为元组返回。其中数组中的每个元素对应一个图像或标签（掩模图像）。
# 这通常是为了将图像和相应的标签整理成一个方便的数据结构，以便后续在模型中使用。
    return np.array( images ), np.array( labels )


# 数据预处理
def preprocess_data(images, labels, target_size=(256, 256)):
    # 调整图像大小
    # 定义一个新的变量列表名image_resized，表示的是上述加载操作中将images列表中的所有图像进行大小调整。
    images_resized = [resize( image, target_size, mode='reflect', anti_aliasing=True ) for image in images]

    # 归一化图像
    # 将图像像素减去min像素除以max像素值与min像素值之差，范围为[0,1]的数值，每个图像都对应一个[0,1]范围内的值，这些数值组成一个名称为images_normalized的列表。
    images_normalized = [(image - np.min( image )) / (np.max( image ) - np.min( image )) for image in images_resized]

    # 归一化标签
    scaler = MinMaxScaler()
    # 创建一个MinMaxScaler对象，即也是使用最大最小缩放的方法，将标签变为[0,1]范围内的数值。
    '''labels_flat = [label.flatten() for label in labels]'''
    # for label in labels是遍历了列表内的所有标签，使用label.flatten()函数将二维数组拉平成一维数组。然后这些一维数组组成了名为labels_flat的列表。
    labels_normalized = [scaler.fit_transform( label.reshape( -1, 1 ) ).reshape( label.shape ) for label in labels]
# scaler.fit_transform(...): 这一步使用 MinMaxScaler 对列向量进行最小-最大缩放。fit_transform 函数会首先计算并保存训练集的最小值和最大值，然后进行相应的缩放操作。
# label.reshape( -1, 1 )将一维数组变成列向量，.reshape( label.shape )函数将进行完最大最小缩放的列向量再变为原来的形状，然后得到得是具有原来形状且进行了缩放的数组，组成labels_normalized列表
    return np.array( images_normalized ), np.array( labels_normalized )
# np.array 函数将 images_normalized 和 labels_normalized 转换为 NumPy 数组，并将它们作为元组返回。其中数组中的每个元素对应一个图像或标签（掩模图像）。
# 这通常是为了将图像和相应的标签整理成一个方便的数据结构，以便后续在模型中使用。


# 定义U-Net模型
# 定义输入的参数为图像大小256*256，三通道的rgb图像。
def unet_model(input_shape=(256, 256, 3)):
    # 使用导入的Input层，为网络结构中的输入层，（）中填入的是定义模型时，所规定的图像参数，此处为256*256，3通道图像。
    inputs = Input( input_shape )

    # 编码器
    conv1 = Conv2D( 64, 3, activation='relu', padding='same' )( inputs )
    # conv1为一个卷积层，是使用Conv2D函数进行创建，其中的参数64 表示是输出的通道数，3表示卷积核的大小此处表示3*3大小的卷积核。
    # activation='relu'表示使用的是relu激活函数，padding='same'表示卷积核滑动时对边缘进行填充，即补0操作，保证输入输出大小相同。
    conv1 = Conv2D( 64, 3, activation='relu', padding='same' )( conv1 )
    # 添加相同配置的卷积层，这种连续添加相同类型的卷积层有助于模型学习图像的复杂特征。
    pool1 = MaxPooling2D( pool_size=(2, 2) )( conv1 )
    # 此处为最大池化层操作，MaxPooling2D函数即构建最大池化层，池化层大小为2*2，输入为conv1.

    conv2 = Conv2D( 128, 3, activation='relu', padding='same' )( pool1 )
    # 紧接上一个卷积层，输入是上一层最后的池化输出，且卷积层输出通道数为128
    conv2 = Conv2D( 128, 3, activation='relu', padding='same' )( conv2 )
    pool2 = MaxPooling2D( pool_size=(2, 2) )( conv2 )

    conv3 = Conv2D( 256, 3, activation='relu', padding='same' )( pool2 )
    # 相同输入为第二层卷积的池化输出，卷积内输出通道数为256.
    conv3 = Conv2D( 256, 3, activation='relu', padding='same' )( conv3 )
    pool3 = MaxPooling2D( pool_size=(2, 2) )( conv3 )

    # 中间层
    # 中间层无池化操作
    conv4 = Conv2D( 512, 3, activation='relu', padding='same' )( pool3 )
    # 卷积层输入为上层池化输出，卷积内的输出通道数为512.
    conv4 = Conv2D( 512, 3, activation='relu', padding='same' )( conv4 )

    # 解码器
    # 上采样层UpSampling2D，concatenate函数
    # UpSampling2D( size=(2, 2) )( conv4 )这里是上采样conv4这个卷积层，并将conv4的特征图放大两倍，然后后面跟conv3
    # 使用concatenate函数将两个张量延轴连接在一起。
    # axis: 指定连接的轴。默认值是 -1，表示沿着最后一个轴进行连接，最后一轴一般为通道轴。
    up5 = concatenate( [UpSampling2D( size=(2, 2) )( conv4 ), conv3], axis=-1 )
    conv5 = Conv2D( 256, 3, activation='relu', padding='same' )( up5 )
    # 卷积层输入为上采样合并的张量，卷积内输出通道数为256
    conv5 = Conv2D( 256, 3, activation='relu', padding='same' )( conv5 )

    up6 = concatenate( [UpSampling2D( size=(2, 2) )( conv5 ), conv2], axis=-1 )
    # 上采样conv5，将特征图放大两倍，与conv2进行延轴链接，得到新张量up6
    conv6 = Conv2D( 128, 3, activation='relu', padding='same' )( up6 )
    # 卷积层输入为up6，卷积内输出通道数为128
    conv6 = Conv2D( 128, 3, activation='relu', padding='same' )( conv6 )

    up7 = concatenate( [UpSampling2D( size=(2, 2) )( conv6 ), conv1], axis=-1 )
    # 上采样conv6将特征图放大两倍，和conv1 延轴链接组成新张量up7
    conv7 = Conv2D( 64, 3, activation='relu', padding='same' )( up7 )
    # 卷积层内输出通道数为64，
    conv7 = Conv2D( 64, 3, activation='relu', padding='same' )( conv7 )

    outputs = Conv2D( 1, 1, activation='sigmoid' )( conv7 )
    # 输出层 输出通道数为1，卷积核大小为1*1，激活函数为sigmoid函数。
    # 输出层 outputs在图像分任务中，最后一层使用 sigmoid 激活函数输出概率值。在图像分割任务中，输出的通道数可能会大于1，表示每个像素属于不同的类别的概率。

    model = Model( inputs=inputs, outputs=outputs )
    # 将输入和输出传递给 Model 类，创建了一个端到端的神经网络模型。
    return model



# 加载和预处理数据
data_directory = "D:\\pycharmyunxing\\iamge"
images, labels = load_data( data_directory )
# 输入图像数据加载路径
images, labels = preprocess_data( images, labels )
# 使用上述加载数据的函数。

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 构建并编译 U-Net 模型
model = unet_model(input_shape=(256, 256, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test))

# 保存训练好的模型权重
model.save_weights("unet_weights.h5")
