import os
import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

# ================= 1. 读取你的数据集 =================
DATA_DIR = "./Digital_Twin_Dataset" # 确保路径对准你有数据的文件夹

clean_files = sorted(glob.glob(os.path.join(DATA_DIR, "Clean_Target", "0152.tif")))
noisy_files = sorted(glob.glob(os.path.join(DATA_DIR, "Noisy_1fps", "0152.tif")))

if len(clean_files) == 0:
    print("❌ 找不到图片，请检查路径！")
    exit()

idx = random.randint(0, len(clean_files) - 1)
print(f"📸 正在使用传统工业算法处理第 {idx+1} 对图像...")

# 读取大图并切出 256 高度的核心区 (保持和你之前一模一样的视野)
clean_img_full = cv2.imread(clean_files[idx], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH).astype(np.float32)
noisy_img_full = cv2.imread(noisy_files[idx], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH).astype(np.float32)

row_sum = np.sum(clean_img_full, axis=1)
brightest_row = int(np.argmax(row_sum))
h, w = clean_img_full.shape
top = max(0, min(h - 256, brightest_row - 128))

clean_img = clean_img_full[top : top + 256, :]
noisy_img = noisy_img_full[top : top + 256, :]

# ================= 2. 核心：传统双重去噪组合拳 =================

# 招式一：2D 空间中值滤波 (彻底抹除底层的独立雪花噪点，窗口大小 3x3)
# 这一步在 2D 图像层面干掉了高频突变噪声
denoised_2d = median_filter(noisy_img, size=3)

# 招式二：沿 X 轴方向的 S-G 滤波 (物理级波峰还原)
# window_length=51 (平滑窗口), polyorder=3 (3次多项式拟合)
# axis=1 表示沿着图像的水平方向(每一行)滑动处理
denoised_2d = savgol_filter(denoised_2d, window_length=51, polyorder=3, axis=1)

# 清除拟合过程中可能产生的极少数负值，保证物理意义正确
denoised_2d = np.clip(denoised_2d, 0, None)

# ================= 3. 数据归一化与可视化 =================
def normalize(img):
    return img / (np.max(img) + 1e-8)

noisy_norm = normalize(noisy_img)
denoised_norm = normalize(denoised_2d)
clean_norm = normalize(clean_img)

def get_1d_spectrum(img_2d):
    spectrum_1d = np.sum(img_2d, axis=0)
    return spectrum_1d / (np.max(spectrum_1d) + 1e-8)

noisy_1d = get_1d_spectrum(noisy_norm)
denoised_1d = get_1d_spectrum(denoised_norm)
clean_1d = get_1d_spectrum(clean_norm)

# 绘制和你之前一样的 2x3 对比图
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Traditional DSP Denoising Test (Image #{idx+1})", fontsize=18, fontweight='bold')

axs[0, 0].imshow(noisy_norm, cmap='gray', vmin=0, vmax=1.0)
axs[0, 0].set_title("1. Input: Noisy (Cropped)")
axs[0, 1].imshow(denoised_norm, cmap='gray', vmin=0, vmax=1.0)
axs[0, 1].set_title("2. DSP Output: Median + S-G Filter")
axs[0, 2].imshow(clean_norm, cmap='gray', vmin=0, vmax=1.0)
axs[0, 2].set_title("3. Ground Truth (Cropped)")

axs[1, 0].plot(noisy_1d, color='red', alpha=0.8)
axs[1, 0].set_title("1D Spectrum: Noisy Input")
axs[1, 1].plot(denoised_1d, color='green', linewidth=2)
axs[1, 1].set_title("1D Spectrum: DSP Reconstructed")
axs[1, 2].plot(clean_1d, color='blue', linewidth=2)
axs[1, 2].set_title("1D Spectrum: Ground Truth")

plt.tight_layout()
plt.show()