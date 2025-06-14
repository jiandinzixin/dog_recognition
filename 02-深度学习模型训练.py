import warnings

warnings.filterwarnings("ignore")
import torch
from PIL import Image
from torchvision import datasets, models, transforms, utils
import torch.nn as nn
import numpy as np
import random
import os
import torchvision
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# 配置日志记录
logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['PYTHONHASHSEED'] = str(seed)


setup_seed(20)

root = './'

# Hyper parameters
num_epochs = 5  # 循环次数
batch_size = 8  # 每次投喂数据量
learning_rate = 0.00005  # 学习率
momentum = 0.9  # 变化率
num_classes = len(os.listdir('./数据集'))  # 几分类


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((line[:-2], int(words[-1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # imgs.append((words[0], int(words[1])))  如果图片的标签类别个数大于等于10用这一行代替上面这一行
            # 很显然，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片 彩色图片则为RGB

        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


# 根据自己定义的那个类MyDataset来创建数据集！注意是数据集！而不是loader迭代器
train_data = MyDataset(datatxt=root + 'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(datatxt=root + 'val.txt', transform=transforms.ToTensor())


print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# 开启gpu加速
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = MobileNet(num_classes).to(device)  # 将模型加载到设备上进行训练

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum = momentum )
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))

train_losses = []
test_losses = []
test_accuracies = []
f1_scores = []

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    train_loader = tqdm(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    net.eval()
    test_loss = 0.
    test_acc = 0.
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = net(batch_x)
            loss2 = criterion(out, batch_y)
            test_loss += loss2.item()
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    test_losses.append(test_loss / len(test_data))
    test_accuracies.append(test_acc / len(test_data))
    f1 = f1_score(all_labels, all_preds, average='weighted')
    f1_scores.append(f1)

    print('Epoch :{}, Test Loss: {:.6f}, Acc: {:.6f}, F1: {:.6f}'.format(epoch + 1, test_loss / (len(
        test_data)), test_acc / (len(test_data)), f1))
    logging.info(
        f'Epoch {epoch + 1}, Test Loss: {(test_loss / len(test_data)):.4f}, Test acc: {(test_acc / len(test_data)):.4f}, F1: {f1:.4f}')

    torch.save(net, 'model.ckpt')

# 创建 result 目录
if not os.path.exists('result'):
    os.makedirs('result')

# 生成指标文本文件
metrics_path = os.path.join('result', 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write("=== 模型训练指标 ===\n")
    f.write(f"训练轮数 (Epochs): {num_epochs}\n")
    f.write(f"类别数量 (Num Classes): {num_classes}\n")
    f.write("-" * 50 + "\n")
    f.write("Epoch\t训练损失\t测试损失\t测试准确率\tF1值\n")
    f.write("-" * 50 + "\n")
    for epoch in range(num_epochs):
        f.write(
            f"{epoch + 1}\t{train_losses[epoch]:.6f}\t{test_losses[epoch]:.6f}\t{test_accuracies[epoch]:.6f}\t{f1_scores[epoch]:.6f}\n")
    f.write("-" * 50 + "\n")
    f.write(f"最高测试准确率: {max(test_accuracies):.6f}\n")
    f.write(f"最高F1值: {max(f1_scores):.6f}\n")

# 绘制训练损失直方图
plt.figure()
plt.bar(range(num_epochs), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss per Epoch')
plt.savefig('result/training_loss_histogram.png')

# 绘制测试损失散点图
plt.figure()
plt.scatter(range(num_epochs), test_losses)
plt.xlabel('Epoch')
plt.savefig('result/test_loss_scatter.png')

# 绘制测试准确率散点图
plt.figure()
plt.scatter(range(num_epochs), test_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy per Epoch')
plt.savefig('result/test_accuracy_scatter.png')

# 绘制 F1 值散点图
plt.figure()
plt.scatter(range(num_epochs), f1_scores)
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score per Epoch')
plt.savefig('result/f1_score_scatter.png')
