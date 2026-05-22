import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# ================= 1. 配置区 =================
DATA_DIR = "./Digital_Twin_Dataset"
BATCH_SIZE = 8       # 如果报显存不足(OOM)，改为 4 或 2
EPOCHS = 150
LEARNING_RATE = 5e-5 # 调低学习率，防止 Loss 震荡
PATCH_H = 64         # 扁长形切块：高度，紧紧包裹光谱线
PATCH_W = 256        # 扁长形切块：宽度，保留足够的光谱特征
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 正在使用计算设备: {DEVICE}")

# ================= 2. 2D 图像数据集加载与智能追踪 =================
class Spectrometer2DDataset(Dataset):
    def __init__(self, data_dir):
        self.clean_files = sorted(glob.glob(os.path.join(data_dir, "Clean_Target", "*.tif")))
        self.noisy_files = sorted(glob.glob(os.path.join(data_dir, "Noisy_1fps", "*.tif")))

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # 读取 16-bit 灰度图
        clean_img = cv2.imread(self.clean_files[idx], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        noisy_img = cv2.imread(self.noisy_files[idx], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH).astype(np.float32)

        # 🔥 核心 1：动态归一化 (根据真实的物理最高亮度)
        max_val = np.max(clean_img) + 1e-8
        clean_img = clean_img / max_val
        noisy_img = noisy_img / max_val
        
        # 防止个别极端亮噪点导致 Noisy 远超 1.0，进行安全截断
        noisy_img = np.clip(noisy_img, 0.0, 1.2)

        h, w = clean_img.shape
        
        # 🔥 核心 2：智能追踪长条切块 (基于干净图像的准确波峰定位)
        row_sum = np.sum(clean_img, axis=1)
        brightest_row = int(np.argmax(row_sum))
        
        # 上下框出 PATCH_H 的高度
        top = brightest_row - PATCH_H // 2
        top = max(0, top)
        top = min(h - PATCH_H, top)
        
        # 水平方向保持随机，让 AI 学到不同波段
        left = random.randint(0, w - PATCH_W)

        clean_patch = clean_img[top : top + PATCH_H, left : left + PATCH_W]
        noisy_patch = noisy_img[top : top + PATCH_H, left : left + PATCH_W]

        # 数据增强：50% 概率翻转
        if random.random() > 0.5:
            clean_patch = np.fliplr(clean_patch)
            noisy_patch = np.fliplr(noisy_patch)
        if random.random() > 0.5:
            clean_patch = np.flipud(clean_patch)
            noisy_patch = np.flipud(noisy_patch)

        # 转换为 PyTorch 格式 [Channels, Height, Width] -> [1, H, W]
        clean_tensor = torch.tensor(clean_patch.copy()).unsqueeze(0)
        noisy_tensor = torch.tensor(noisy_patch.copy()).unsqueeze(0)

        return noisy_tensor, clean_tensor

# ================= 3. 标准 2D U-Net 网络结构 =================
class UNet2D(nn.Module):
    def __init__(self):
        super(UNet2D, self).__init__()
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.down1 = double_conv(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = double_conv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = double_conv(128, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = double_conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = double_conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up1 = double_conv(64, 32)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        b = self.bottleneck(self.pool3(d3))
        u3 = self.conv_up3(torch.cat([self.up3(b), d3], dim=1))
        u2 = self.conv_up2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv_up1(torch.cat([self.up1(u2), d1], dim=1))
        return self.out_conv(u1)

# ================= 4. 训练引擎 =================
def train():
    dataset = Spectrometer2DDataset(DATA_DIR)
    
    # === 替换原来的 random_split ===
    train_size = len(dataset) - 100
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, 100])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    # 验证集的 batch_size 可以设为 4，加快验证速度
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    
    # ============================

    model = UNet2D().to(DEVICE)
    import os
    MODEL_PATH = "best_2d_denoise_model.pth"
    if os.path.exists(MODEL_PATH):
        print(f"🔄 检测到历史模型 {MODEL_PATH}，正在加载以继续训练...")
        # 记得加上 weights_only=True 避免之前的警告
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    else:
        print("🆕 未检测到历史模型，初始化全新权重开始训练...")
    # 🔥 核心 3：换成 MSELoss，狠狠惩罚网络漏掉微弱波峰的行为
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')

    # === 彻底替换整个 epoch 循环 ===
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0  # ✅ 修复的 Bug：每轮开始前必须清零！
            
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(noisy)
                
            # 🔥 训练时的特征加权 Loss
            mask = 1.0 + 20.0 * clean
            
            loss_weighted = ((outputs - clean) ** 2) * mask
            loss = loss_weighted.mean()
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
                
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                outputs = model(noisy)
                    
                # 🔥 验证时也用同样的加权标准！不让它骗我们！
                val_mask = 1.0 + 20.0 * clean
                
                val_loss_weighted = ((outputs - clean) ** 2) * val_mask
                    
                val_loss += val_loss_weighted.mean().item()
                    
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | 加权 Train MSE: {avg_train:.6f} | 加权 Val MSE: {avg_val:.6f}")
            
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), "best_2d_denoise_model.pth")
            print("⭐ 最佳加权模型已保存！")
    # ============================================

if __name__ == "__main__":
    train()