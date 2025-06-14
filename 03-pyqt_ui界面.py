# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore")
import sys  # 导入系统
from PIL import Image
import torch.nn as nn
from torchvision import transforms, models
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QTextEdit, QFileDialog, \
    QHBoxLayout, QVBoxLayout, QSplitter, QComboBox, QSpinBox
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5 import QtCore, QtGui
import numpy as np
import torch
import os
import torch
import torch.nn as nn
import argparse
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # 定义字典className_list，把种类索引转换为种类名称
classes = os.listdir('./数据集')

# ------------------------------------------------------1.加载模型--------------------------------------------------------------
num_classes = len(classes)


class MobileNet(nn.Module):
    def __init__(self, num_classes=num_classes):  # num_classes
        super(MobileNet, self).__init__()
        net = models.mobilenet_v2(pretrained=True)  # 从预训练模型加载mobilenet_v2网络参数
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(1280, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ResNet18, self).__init__()
        net = models.resnet18(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


path_model1 = "./model.ckpt"
model1 = torch.load(path_model1)
model1 = model1.to(device)

path_model2 = "./model2.ckpt"
model2 = torch.load(path_model2)
model2 = model2.to(device)


# 根据图片文件路径获取图像数据矩阵
def get_imageNdarray(imageFilePath):
    input_image = Image.open(imageFilePath).convert("RGB")
    return input_image


# 模型预测前必要的图像处理
def process_imageNdarray(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_chw = preprocess(input_image)
    return img_chw  # chw:channel height width


class FirstUi(QMainWindow):  # 第一个窗口类
    def __init__(self):
        super(FirstUi, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(self.width(), self.height())
        self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(700, 500)  # 设置窗口大小
        self.btn2 = QPushButton('图片识别', self)
        self.btn2.setGeometry(245, 200, 150, 50)
        self.btn2.clicked.connect(self.slot_btn2_function)
        self.btn_exit = QPushButton('退出', self)
        self.btn_exit.setGeometry(245, 300, 150, 50)
        self.btn_exit.clicked.connect(self.Quit)
        self.label_name = QLabel('welcome 图片识别', self)
        self.label_name.setGeometry(460, 410, 200, 30)

    def Quit(self):
        self.close()

    def slot_btn_function(self):
        self.hide()  # 隐藏此窗口
        self.s = write_num()  # 将第二个窗口换个名字
        self.s.show()  # 经第二个窗口显示出来

    def slot_btn2_function(self):
        self.hide()  # 隐藏此窗口
        self.s = picture_num()
        self.s.show()

    def __fillColorList(self, comboBox):
        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()

    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return

    def Quit(self):
        self.close()


class picture_num(QWidget):
    def __init__(self):
        super(picture_num, self).__init__()
        self.init_ui()
        self.fname = ''

    def init_ui(self):
        self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(520, 540)
        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle('图片识别')

        self.label_name5 = QLabel('待载入图片', self)
        self.label_name5.setGeometry(10, 20, 500, 380)
        self.label_name5.setStyleSheet("QLabel{background:gray;}"
                                       "QLabel{color:rgb(0,0,0,120);font-size:15px;font-weight:bold;font-family:宋体;}"
                                       )
        self.label_name5.setAlignment(QtCore.Qt.AlignCenter)

        self.edit = QTextEdit(self)
        self.edit.setGeometry(340, 420, 170, 80)
        self.edit.setFont(QtGui.QFont("Arial", 14, weight=QtGui.QFont.Bold))

        self.btn_select = QPushButton('选择图片', self)
        self.btn_select.setGeometry(20, 420, 100, 30)
        self.btn_select.clicked.connect(self.select_image)

        self.btn_dis_model1 = QPushButton('MobileNet V2模型识别图片', self)
        self.btn_dis_model1.setGeometry(20, 460, 150, 30)
        self.btn_dis_model1.clicked.connect(self.on_btn_Recognize_Model1_Clicked)

        self.btn_dis_model2 = QPushButton('ResNet-18模型识别图片', self)
        self.btn_dis_model2.setGeometry(180, 460, 150, 30)
        self.btn_dis_model2.clicked.connect(self.on_btn_Recognize_Model2_Clicked)

        self.btn = QPushButton('返回', self)
        self.btn.setGeometry(20, 500, 100, 30)
        self.btn.clicked.connect(self.slot_btn_function)

        self.btn_exit = QPushButton('退出', self)
        self.btn_exit.setGeometry(160, 500, 100, 30)
        self.btn_exit.clicked.connect(self.Quit)

    def select_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label_name5.width(), self.label_name5.height())
        self.label_name5.setPixmap(jpg)
        self.fname = imgName

    def on_btn_Recognize_Model1_Clicked(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        savePath = self.fname
        input_image = get_imageNdarray(savePath)  # 打开单张图片
        input_image = input_image.resize((224, 224))
        img_chw = process_imageNdarray(input_image)

        if torch.cuda.is_available():
            img_chw = img_chw.view(1, 3, 224, 224).to(device)
        else:
            img_chw = img_chw.view(1, 3, 224, 224)
        model1.eval()
        with torch.no_grad():
            out = model1(img_chw)
            score = torch.nn.functional.softmax(out, dim=1)[0] * 100  # 获得所有目标的准确率
            predicted = torch.max(out, 1)[1]
            score = score[predicted.item()].item()  # 获得最大目标的准确率
            txt = str(classes[predicted.item()])
        self.edit.setText(str(txt))

    def on_btn_Recognize_Model2_Clicked(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        savePath = self.fname
        input_image = get_imageNdarray(savePath)  # 打开单张图片
        input_image = input_image.resize((224, 224))
        img_chw = process_imageNdarray(input_image)

        if torch.cuda.is_available():
            img_chw = img_chw.view(1, 3, 224, 224).to(device)
        else:
            img_chw = img_chw.view(1, 3, 224, 224)
        model2.eval()
        with torch.no_grad():
            out = model2(img_chw)
            score = torch.nn.functional.softmax(out, dim=1)[0] * 100  # 获得所有目标的准确率
            predicted = torch.max(out, 1)[1]
            score = score[predicted.item()].item()  # 获得最大目标的准确率
            txt = str(classes[predicted.item()])
        self.edit.setText(str(txt))

    def Quit(self):
        self.close()

    def slot_btn_function(self):
        self.hide()
        self.f = FirstUi()
        self.f.show()


def main():
    app = QApplication(sys.argv)
    w = FirstUi()  # 将第一和窗口换个名字
    w.show()  # 将第一和窗口换个名字显示出来
    sys.exit(app.exec_())  # app.exet_()是指程序一直循环运行直到主窗口被关闭终止进程（如果没有这句话，程序运行时会一闪而过）


if __name__ == '__main__':  # 只有在本py文件中才能用，被调用就不执行
    main()
