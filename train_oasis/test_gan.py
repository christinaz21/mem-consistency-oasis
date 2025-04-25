import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # 添加 tqdm 导入
from torchvision.utils import save_image  # 添加 save_image 导入
import os  # 添加 os 导入

import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
import torch.nn.functional as F

# Hyperparameters
batch_size = 256
epochs = 200
lr = 1e-5
beta1 = 0.5

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist_data = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Generator
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.label_embedding = nn.Embedding(10, 28 * 28)  # 嵌入层用于条件

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1 + 1, 64, kernel_size=4, stride=2, padding=1),  # 输入图像和条件
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # 嵌入条件并扩展维度
        c = self.label_embedding(labels)
        c = c.view(c.size(0), 1, 28, 28)  # 将条件嵌入转换为图像形状

        x = torch.cat([z, c], dim=1)

        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 解码
        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return d1

# Discriminator
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(10, 28 * 28)  # 嵌入层用于条件
        self.initial = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.resnet_blocks = nn.Sequential(
            ResNetBlock(64, 64, stride=1),
            ResNetBlock(64, 64, stride=1),
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 128, stride=1),
            ResNetBlock(128, 128, stride=1),
            ResNetBlock(128, 128, stride=1),
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 256, stride=1),
            ResNetBlock(256, 256, stride=1),
            ResNetBlock(256, 256, stride=1),
            ResNetBlock(256, 256, stride=1),
            ResNetBlock(256, 256, stride=1),
            ResNetBlock(256, 512, stride=2),
            ResNetBlock(512, 512, stride=1),
            ResNetBlock(512, 512, stride=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels).view(labels.size(0), 1, 28, 28)  # 嵌入条件
        x = torch.cat([x, c], dim=1)  # 将图像和条件拼接
        x = self.initial(x)
        x = self.resnet_blocks(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.fc(x)

# Initialize models
generator = UNetGenerator().to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# 创建保存生成图片的目录
save_dir = "outputs/test_gan"
if os.path.exists(save_dir):
    import shutil
    shutil.rmtree(save_dir)  # 删除旧目录
os.makedirs(save_dir, exist_ok=True)

# Training loop
for epoch in range(epochs):
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs}")
    for i, (real_images, real_labels) in progress_bar:
        real_images, real_labels = real_images.to(device), real_labels.to(device)
        batch_size = real_images.size(0)

        # Labels
        real_targets = torch.ones(batch_size, 1).to(device)
        fake_targets = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        optimizer_d.zero_grad()
        real_outputs = discriminator(real_images, real_labels)
        real_loss = criterion(real_outputs, real_targets)

        z = torch.randn_like(real_images).to(device)  # 随机噪声
        fake_labels = torch.randint(0, 10, (batch_size,)).to(device)  # 随机生成条件
        fake_images = generator(z, fake_labels)
        fake_outputs = discriminator(fake_images.detach(), fake_labels)
        fake_loss = criterion(fake_outputs, fake_targets)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        clip_grad_value_(discriminator.parameters(), clip_value=1.0)
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        fake_outputs = discriminator(fake_images, fake_labels)
        g_loss = criterion(fake_outputs, real_targets)
        g_loss.backward()
        clip_grad_value_(generator.parameters(), clip_value=1.0)
        optimizer_g.step()

        # 更新 tqdm 描述信息
        progress_bar.set_postfix(D_Loss=d_loss.item(), G_Loss=g_loss.item())

    # 保存生成的图片
    if epoch % 10 == 0:
        with torch.no_grad():
            z = torch.randn(32, 1, 28, 28).to(device)
            labels = torch.arange(0, 10).repeat(4, 1).view(-1).to(device)[:32]  # 生成条件
            fake_images = generator(z, labels)
            save_image(fake_images * -1 + 1, os.path.join(save_dir, f"epoch-{epoch}.png"), nrow=8)