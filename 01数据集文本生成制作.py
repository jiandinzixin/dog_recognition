import os
import random

#设置随机种子 确保对图片每次打乱顺序都是一样的
def setup_seed(seed):
    random.seed(seed)
setup_seed(20)

b = 0
dir = './数据集/'
#os.listdir的结果就是一个list集合
#可以使用一个list的sort方法进行排序，有数字就用数字排序
a = 0
a1 = 0
files = os.listdir('./数据集')
while(b < len(files)):#这里是分类个数
    label = a #设置要标记的标签
    ss = './数据集/' + str(files[b]) + '/' #训练图片
    print(f"正在处理分类: {files[b]}，对应的标签为: {label}")
    pics = os.listdir(ss) #得到sample00_train文件夹下的图片
    i = 1
    train_percent = 0.8  # 训练集样本占比 训练集0.8 则测试集0.2

    num = len(pics)  # 得到样本总数
    list = range(num)  #得到列表
    train_num = int(num * train_percent)  # 训练集总数
    test_num = num - train_num     #获得测试样本数

    train_sample = random.sample(list, train_num)  # 在list中随机选择 train_num个长度，并乱序

    for i in list:  # 循环输出文件
        name = str(dir) + str(files[b]) + '/' + pics[i] + ' ' + str(int(label)) + '\n'  # 获得当前文件夹下所有图片序列名称
        if i in train_sample:  # 判断i是否在训练集中
            train.write(name)  # 如果在，输出图片做训练文本中
        else:
            test.write(name)    #其余的做测试文本中
    a = a + 1
    b = b + 1
train.close()  #操作完成后一定要记得关闭文件
test.close()