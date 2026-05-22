import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# ================= 1. 配置区 =================
DATA_DIR = "./Digital_Twin_Dataset"
MODEL_PATH = "best_2d_denoise_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 2. 引入 2D U-Net 模型架构 (必须与训练时完全一致) =================
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
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        b = self.bottleneck(self.pool3(d3))
        u3 = self.conv_up3(torch.cat([self.up3(b), d3], dim=1))
        u2 = self.conv_up2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv_up1(torch.cat([self.up1(u2), d1], dim=1))
        return self.out_conv(u1)

# ================= 3. 加载模型与数据准备 =================
print("🧠 正在唤醒 AI 模型...")
model = UNet2D().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # 开启评估模式，极其重要！
    print("✅ 模型加载成功！")
else:
    print(f"❌ 找不到模型文件 {MODEL_PATH}")
    exit()

clean_files = sorted(glob.glob(os.path.join(DATA_DIR, "Clean_Target", "*.tif")))
noisy_files = sorted(glob.glob(os.path.join(DATA_DIR, "Noisy_1fps", "*.tif")))

# 随机抽一张图来测试 (你也可以把 idx 写死，比如 idx = 10 来看特定的图)
idx = random.randint(0, len(clean_files) - 1)
print(f"📸 正在测试第 {idx+1} 对图像...")

# 读取图像
clean_img = cv2.imread(clean_files[idx], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH).astype(np.float32)
noisy_img = cv2.imread(noisy_files[idx], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH).astype(np.float32)

# 记录原始最大值用于显示
c_max_raw = np.max(clean_img)
n_max_raw = np.max(noisy_img)

# 按照训练时的逻辑进行独立归一化
clean_norm = clean_img / (c_max_raw + 1e-8)
noisy_norm = noisy_img / (n_max_raw + 1e-8)
noisy_norm = np.clip(noisy_norm, 0.0, 1.2)

# ================= 4. AI 推理执行 =================
# 转换为 Tensor: [1, 1, H, W]
tensor_in = torch.tensor(noisy_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    tensor_out = model(tensor_in)

# 将 AI 输出转回 numpy 数组，并去负值保护
ai_out_norm = tensor_out.squeeze().cpu().numpy()
ai_out_norm = np.clip(ai_out_norm, 0.0, 1.0)

# ================= 5. 可视化终极对比大屏 =================
# 为了 1D 曲线的对比，我们将它们都压缩成一维并归一化到 0-1 方便同框比较
def get_1d_spectrum(img_2d):
    spectrum_1d = np.sum(img_2d, axis=0)
    return spectrum_1d / (np.max(spectrum_1d) + 1e-8)

noisy_1d = get_1d_spectrum(noisy_norm)
ai_1d = get_1d_spectrum(ai_out_norm)
clean_1d = get_1d_spectrum(clean_norm)

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"AI Denoising Performance Test (Image #{idx+1})", fontsize=18, fontweight='bold')

# --- 第一排：2D 图像直观对比 ---
axs[0, 0].imshow(noisy_norm, cmap='gray', vmin=0, vmax=1.0)
axs[0, 0].set_title("1. Input: Noisy (Short Exposure)")
axs[0, 0].axis('off')

axs[0, 1].imshow(ai_out_norm, cmap='gray', vmin=0, vmax=1.0)
axs[0, 1].set_title("2. AI Output: Denoised")
axs[0, 1].axis('off')

axs[0, 2].imshow(clean_norm, cmap='gray', vmin=0, vmax=1.0)
axs[0, 2].set_title("3. Ground Truth: Clean (Long Exposure)")
axs[0, 2].axis('off')

# --- 第二排：1D 光谱曲线精准对比 ---
axs[1, 0].plot(noisy_1d, color='red', alpha=0.8)
axs[1, 0].set_title("1D Spectrum: Noisy Input")
axs[1, 0].grid(True, linestyle='--', alpha=0.5)

axs[1, 1].plot(ai_1d, color='green', linewidth=2)
axs[1, 1].set_title("1D Spectrum: AI Reconstructed")
axs[1, 1].grid(True, linestyle='--', alpha=0.5)

axs[1, 2].plot(clean_1d, color='blue', linewidth=2)
axs[1, 2].set_title("1D Spectrum: Ground Truth")
axs[1, 2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()