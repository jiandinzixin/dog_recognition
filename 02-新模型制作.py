import warnings

warnings.filterwarnings("ignore")
import torch
from PIL import Image

import torch.nn as nn
import numpy as np
import random


# 配置日志记录
logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    os.environ['PYTHONHASHSEED'] = str(seed)


setup_seed(20)

root = './'

# Hyper parameters
num_epochs = 5  # 循环次数

momentum = 0.9  # 变化率
num_classes = len(os.listdir('./数据集'))  # 自动获取犬种数量


# 自定义数据集类（保留你的原始数据加载方式）
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        img = img.resize((224, 224))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


# 加载数据集（通过txt文件）
train_data = MyDataset(datatxt=root + 'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(datatxt=root + 'val.txt', transform=transforms.ToTensor())

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# 设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 模型定义
class ResNet18(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ResNet18, self).__init__()
        net = models.resnet18(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Sequential()
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


net = ResNet18(num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))

train_losses = []
test_losses = []
test_accuracies = []
f1_scores = []

# 训练循环
for epoch in range(num_epochs):
    running_loss = 0.0
    train_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # 测试阶段
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = net(batch_x)
            loss2 = criterion(out, batch_y)

            test_loss += loss2.item()
            pred = torch.max(out, 1)[1]
            test_acc += (pred == batch_y).sum().item()
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    test_loss_per_epoch = test_loss / len(test_data)
    test_accuracy = test_acc / len(test_data)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    test_losses.append(test_loss_per_epoch)
    test_accuracies.append(test_accuracy)
    f1_scores.append(f1)

    # 输出到控制台
    print('Epoch :{}, Test Loss: {:.6f}, Acc: {:.6f}, F1: {:.6f}'.format(
        epoch + 1, test_loss_per_epoch, test_accuracy, f1
    ))

    # 记录到日志
    logging.info(
        f'Epoch {epoch + 1}, Test Loss: {test_loss_per_epoch:.4f}, '
        f'Test acc: {test_accuracy:.4f}, F1: {f1:.4f}'
    )

    # 保存模型（按Epoch覆盖，如需保留所有Epoch模型，可改为model_epoch_{epoch+1}.ckpt）
    torch.save(net, 'model2.ckpt')

# 创建结果目录
result_dir = 'result2'
os.makedirs(result_dir, exist_ok=True)

# ----------------------
# 生成指标文本文件（修改部分）
# ----------------------
metrics_path = os.path.join(result_dir, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write("=== 模型训练指标 ===\n")
    f.write(f"训练轮数 (Epochs): {num_epochs}\n")
    f.write(f"批量大小 (Batch Size): {batch_size}\n")
    f.write(f"学习率 (Learning Rate): {learning_rate:.0e}\n")
    f.write(f"类别数量 (Num Classes): {num_classes}\n")
    f.write("-" * 50 + "\n")
    f.write("Epoch\t训练损失\t测试损失\t测试准确率\tF1值\n")
    f.write("-" * 50 + "\n")

    for idx in range(num_epochs):
        train_loss = train_losses[idx]
        test_loss = test_losses[idx]
        test_acc = test_accuracies[idx]
        f1_val = f1_scores[idx]
        f.write(f"{idx + 1}\t{train_loss:.6f}\t{test_loss:.6f}\t{test_acc:.6f}\t{f1_val:.6f}\n")

    f.write("-" * 50 + "\n")
    f.write(f"最高测试准确率: {max(test_accuracies):.6f}\n")
    f.write(f"最高F1值: {max(f1_scores):.6f}\n")

# ----------------------
# 绘制训练损失直方图
plt.figure()
plt.bar(range(num_epochs), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss per Epoch')
plt.savefig(os.path.join(result_dir, 'training_loss_histogram.png'))

# 绘制测试损失散点图
plt.figure()
plt.scatter(range(num_epochs), test_losses)
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss per Epoch')
plt.savefig(os.path.join(result_dir, 'test_loss_scatter.png'))

# 绘制测试准确率散点图
plt.figure()
plt.scatter(range(num_epochs), test_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy per Epoch')
plt.savefig(os.path.join(result_dir, 'test_accuracy_scatter.png'))

# 绘制 F1 值散点图
plt.figure()
plt.scatter(range(num_epochs), f1_scores)
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score per Epoch')
plt.savefig(os.path.join(result_dir, 'f1_score_scatter.png'))

print(f"\n所有结果已保存至 {result_dir} 目录")
