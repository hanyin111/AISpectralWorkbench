import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ================= 1. 你的“神级架构” (保持不变) =================
class PeakFinderSoftArgmax(nn.Module):
    def __init__(self, seq_len):
        super(PeakFinderSoftArgmax, self).__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, padding=14, dilation=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=15, padding=28, dilation=4),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1)
        )
        self.register_buffer('indices', torch.arange(seq_len, dtype=torch.float32))

    def forward(self, x):
        x = x.unsqueeze(1)
        logits = self.net(x).squeeze(1)
        weights = torch.softmax(logits, dim=-1)
        peak_pos = torch.sum(weights * self.indices, dim=-1)
        return peak_pos, weights

# ================= 2. 动态标定公式 (保持不变) =================
def get_calibration_params(target_center_wl):
    """你推导的全局抛物线公式，请确保 A 和 B 的系数是最新的"""
    A = (1.88082617e-08) * (target_center_wl ** 2) + (-2.14228025e-05) * target_center_wl + (1.54531419e-02)
    B = (-9.94596779e-05) * (target_center_wl ** 2) + (1.09995751e+00) * target_center_wl + (-3.37890078e+01)
    return A, B

# ================= 3. 终极实战与真理对比引擎 =================
def analyze_real_image_with_zolix(image_path, zolix_txt_path, model, device, center_wl_setting):
    """同时加载 CCD 图像和 Zolix 扫描数据进行绝对对比"""
    
    # ------------------ A. 处理真实的 CCD 图像 ------------------
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH).astype(np.float32)
    if img is None:
        print(f"❌ 找不到图片 {image_path}")
        return
        
    H, W = img.shape
    row_sum = np.sum(img, axis=1)
    brightest_row = int(np.argmax(row_sum))
    top = max(0, min(H - 256, brightest_row - 128))
    core_img = img[top : top + 256, :]
    raw_1d_signal = np.sum(core_img, axis=0)
    
    # CCD 归一化
    signal_min = np.min(raw_1d_signal)
    signal_max = np.max(raw_1d_signal)
    ai_input_signal = (raw_1d_signal - signal_min) / (signal_max - signal_min + 1e-8)
    
    # ------------------ B. AI 大脑提取像素坐标 ------------------
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(ai_input_signal, dtype=torch.float32).unsqueeze(0).to(device)
        predicted_pixel, _ = model(input_tensor)
        predicted_pixel = predicted_pixel.item()
        
    # 物理引擎：像素 -> 纳米
    A, B = get_calibration_params(center_wl_setting)
    predicted_wavelength = predicted_pixel * A + B
    
    # ------------------ C. 加载并处理 Zolix 真理数据 (10nm 极限视野版) ------------------
    zolix_wl, zolix_int = None, None
    if os.path.exists(zolix_txt_path):
        try:
            zolix_data = np.loadtxt(zolix_txt_path)
            global_wl = zolix_data[:, 0]
            global_raw_int = zolix_data[:, 1]
            
            # 【终极修复：10nm 极限视场切割 (FOV Cropping)】
            # 既然照片是在波峰正中心附近拍的，直接把搜索半径锁死在 10nm！
            # 彻底屏蔽掉几十纳米外那些极其明亮的干扰谱线
            search_radius = 10.0 
            fov_mask = (global_wl >= center_wl_setting - search_radius) & (global_wl <= center_wl_setting + search_radius)
            
            zolix_wl = global_wl[fov_mask]
            zolix_raw_int = global_raw_int[fov_mask]
            
            if len(zolix_wl) > 0:
                # 【局部归一化】让当前狭窄视场内的小波峰也能长到 1.0 的高度
                z_min = np.min(zolix_raw_int)
                z_max = np.max(zolix_raw_int)
                zolix_int = (zolix_raw_int - z_min) / (z_max - z_min + 1e-8)
                
                # 【局部寻峰】此时视野里只有一个峰，绝对不可能跑偏了！
                true_zolix_peak = zolix_wl[np.argmax(zolix_int)]
                print(f"✔️ Zolix 极限局部截取成功！完美锁定目标基准: {true_zolix_peak:.3f} nm")
            else:
                print(f"⚠️ 警告：Zolix 数据中不包含 {center_wl_setting} nm 附近 +/- 10nm 的数据段！")
                
        except Exception as e:
            print(f"⚠️ Zolix 数据加载失败: {e}")
    else:
        print(f"⚠️ 找不到 Zolix 数据文件: {zolix_txt_path}")

    # ------------------ D. 绘制终极大结局图表 ------------------
    plt.figure(figsize=(14, 7))
    
    # CCD 图像的 X 轴物理映射
    x_wavelengths = np.arange(W) * A + B
    
    # 1. 画 CCD 真实原始信号 (胖胖的包络)
    plt.plot(x_wavelengths, ai_input_signal, color='gray', linewidth=3, alpha=0.5, label=f"Raw CCD Signal (Instant)")
    
    # 2. 画 Zolix 真理信号 (极细的基准)
    if zolix_wl is not None:
        plt.plot(zolix_wl, zolix_int, color='blue', linewidth=2, alpha=0.8, label="Zolix PMT Ground Truth (Slow Scan)")
        
        # 寻找 Zolix 的绝对最强峰作为真值对比
        true_zolix_peak = zolix_wl[np.argmax(zolix_int)]
        plt.axvline(x=true_zolix_peak, color='cyan', linestyle=':', linewidth=2, 
                    label=f"Zolix Peak: {true_zolix_peak:.3f} nm")
        
        # 计算极其震撼的物理误差！
        phys_error_nm = abs(predicted_wavelength - true_zolix_peak)
    
    # 3. 画 AI 推理的绝对波长线
    plt.axvline(x=predicted_wavelength, color='red', linestyle='--', linewidth=2, 
                label=f"AI Calibrated Peak: {predicted_wavelength:.3f} nm")
    
    plt.title(f"AI Super-Resolution Calibration vs Physical Ground Truth", fontweight='bold', fontsize=16)
    plt.xlabel("Absolute Wavelength (nm)", fontsize=14)
    plt.ylabel("Normalized Intensity (a.u.)", fontsize=14)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # 自动放大视野：锁定在 AI 预测波长的左右 5 nm 范围内
    plt.xlim(predicted_wavelength - 5, predicted_wavelength + 5)
    
    # 在图表空白处打上“水印”般的测量报告
    report_text = (
        f"AI Sub-pixel: {predicted_pixel:.3f} px\n"
        f"Dispersion (A): {A:.5f} nm/px\n"
        f"AI Wavelength: {predicted_wavelength:.3f} nm"
    )
    if zolix_wl is not None:
        report_text += f"\nZolix True WL: {true_zolix_peak:.3f} nm\nAbsolute Error: {phys_error_nm:.4f} nm"
        
    plt.text(0.02, 0.95, report_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CCD_PIXEL_WIDTH = 2048 
    
    model = PeakFinderSoftArgmax(CCD_PIXEL_WIDTH).to(device)
    model.load_state_dict(torch.load("models/peak_finder_argmax.pth", weights_only=True))
    
    # 【请在这里填入你的文件路径】
    # 确保 0.1step.txt 是两列数据（波长，强度），没有文字表头。
    analyze_real_image_with_zolix(
        image_path="541nm.tif",       # 你的真实相机照片
        zolix_txt_path="0.1step.txt",     # Zolix 慢扫描真值数据
        model=model, 
        device=device, 
        center_wl_setting=541.0           # 当时的中心波长设定值
    )