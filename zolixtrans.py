import numpy as np
from scipy.signal import find_peaks

# 读取你的高精度数据
filepath = "0.1step.txt"
data = np.genfromtxt(filepath, skip_header=1, invalid_raise=False)
wavelengths = data[:, 0]
intensities = data[:, 1]

# 找到前 6 个最强的峰
peaks, properties = find_peaks(intensities, prominence=np.max(intensities)*0.1)
peak_intensities = intensities[peaks]

# 按强度从高到低排序输出
sorted_indices = np.argsort(peak_intensities)[::-1]
print(f"🎯 0.1nm 扫描文件 {filepath} 中的最强特征峰：")
print("-" * 40)
for i in sorted_indices[:6]:
    wl = wavelengths[peaks[i]]
    intensity = intensities[peaks[i]]
    print(f"Zolix 记录波长: {wl:.2f} nm  |  强度: {intensity:.1e}")
print("-" * 40)