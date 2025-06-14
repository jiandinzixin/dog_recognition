# dog_recognition
使用数据集训练模型后，对图片中的犬类进行识别

综述：

后端实现使用Python 语言，基于PyTorch框架进行深度学习模型开发，利用其动态计算图机制和自动微分功能实现神经网络的搭建与优化，结合 MobileNet V2和ResNet-18预训练模型完成犬类图像的特征提取与分类。界面交互部分采用PyQt5库开发图形用户界面，使用信号与槽机制实现图像上传、模型切换及结果展示等功能。数据处理环节通过自定义MyDataset类和PyTorch的 DataLoader实现数据集的加载与批量处理，并使用Matplotlib进行训练过程的可视化。系统在Intel i7-10700K CPU、NVIDIA RTX 3060 GPU（CUDA 11.3）的硬件环境下进行模型训练，移动端推理则适配ARM架构芯片的低算力环境。系统开发中使用到的开发工具、开发环境详细描述如下表所示：

![1749872307187](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\1749872307187.png)

更新日志：

250609上传犬类图片数据集

250610数据集文本代码测试使用

250611模型制作代码

250614-001

开发完剩余代码模块进行上传，程序运行图如下

![1749872156067](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\1749872156067.png)

![1749872170080](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\1749872170080.png)

![1749872188245](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\1749872188245.png)

250614-002

增加文档前面部分综述





