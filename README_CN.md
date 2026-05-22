# AI 光谱分析工作台

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)](#)

AI 光谱分析工作台是一个面向光致发光光谱数据处理的 Windows 桌面可视化程序。项目集成了传统 DSP 去噪、AI 二维 U-Net 去噪、AI 亚像素寻峰和 Zolix TXT/CSV 光谱数据可视化转换功能，适合用于弱光光谱图像分析、实验结果展示和算法对比。

英文说明请见 [README.md](README.md)。

## 项目简介

本项目面向弱光光谱检测场景，尤其适合处理 CCD 光谱图像和 Zolix 光谱仪慢扫描数据。程序将多个原本分散的算法脚本整合到一个图形界面中，用户可以通过鼠标选择功能、导入图像或 TXT 文件、选择模型权重，并在界面中直接查看处理结果。

## GitHub 项目描述

中文 description：

```text
一个用于光致发光光谱图像处理的 Windows 桌面可视化工具，集成 DSP/AI 去噪、AI 亚像素寻峰和 Zolix TXT/CSV 光谱曲线转换功能。
```

## 主要功能

- **光谱图像去噪**
  - DSP 去噪：使用中值滤波和 Savitzky-Golay 平滑进行传统信号处理。
  - AI 去噪：使用二维 U-Net 模型 `best_2d_denoise_model.pth` 对低信噪比光谱图像进行重建。
  - 支持显示二维图像对比和一维光谱曲线。
  - 支持可选参考真值图像，并计算 RMSE 和 PSNR。

- **AI 亚像素寻峰**
  - 读取 CCD 光谱图像。
  - 自动裁剪最亮的核心光谱区域。
  - 使用 `peak_finder_argmax.pth` 预测亚像素峰位。
  - 通过动态定标模型将像素坐标转换为物理波长。
  - 可选加载 Zolix 慢扫描 TXT/CSV 文件进行真值对比。

- **Zolix TXT/CSV 可视化转换**
  - 读取两列波长-强度数据。
  - 绘制光谱曲线。
  - 自动识别并标注主要特征峰。

- **桌面可视化界面**
  - 基于 Tkinter 和 Matplotlib。
  - 所有核心操作都可以通过鼠标点击完成。
  - 结果图可以保存为 PNG 或 PDF。

## 项目文件说明

| 文件 | 说明 |
| --- | --- |
| `spectral_analysis_gui.py` | 主 GUI 程序 |
| `run_gui.bat` | Windows 源码运行脚本 |
| `build_exe.bat` | PyInstaller 打包脚本 |
| `requirements.txt` | 运行依赖 |
| `requirements-dev.txt` | 运行依赖和打包依赖 |
| `best_2d_denoise_model.pth` | AI 去噪模型权重 |
| `peak_finder_argmax.pth` | AI 寻峰模型权重 |
| `traditional.py` | 原 DSP 去噪演示脚本 |
| `test_2d_model.py` | 原 AI 去噪演示脚本 |
| `model.py` | 原 AI 寻峰演示脚本 |
| `zolixtrans.py` | 原 Zolix TXT 寻峰演示脚本 |
| `INSTALL.md` | 面向新手的详细安装说明 |
| `LICENSE` | MIT 开源许可证 |

## 快速开始

### 方式一：源码运行

请先安装 Python 3.11 或更高版本，然后在项目目录中运行：

```bat
python -m pip install -r requirements.txt
python spectral_analysis_gui.py
```

Windows 用户也可以直接双击：

```text
run_gui.bat
```

### 方式二：打包为 exe

安装开发依赖：

```bat
python -m pip install -r requirements-dev.txt
```

执行打包：

```bat
python -m PyInstaller --noconfirm --clean --windowed --onefile --name AISpectralWorkbench --add-data "best_2d_denoise_model.pth;." --add-data "peak_finder_argmax.pth;." spectral_analysis_gui.py
```

也可以直接双击：

```text
build_exe.bat
```

生成的 exe 文件位于：

```text
dist\AISpectralWorkbench.exe
```

注意：由于程序包含 PyTorch、OpenCV、Matplotlib 和模型权重，打包后的 exe 文件体积会比较大。

## 输入数据格式

### 光谱图像

支持以下格式：

- `.tif`
- `.tiff`
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`

推荐使用实验采集得到的原始 CCD 光谱图像，尤其是 `.tif` 格式。

### Zolix TXT/CSV 文件

文件至少需要包含两列数值：波长和强度。

空格或制表符分隔示例：

```text
wavelength intensity
300.0 120
300.1 125
300.2 130
```

逗号分隔示例：

```text
wavelength,intensity
300.0,120
300.1,125
300.2,130
```

## 使用方法

1. 启动程序。
2. 在主页选择功能：
   - `光谱图像去噪`
   - `AI 亚像素寻峰`
   - `图像转换`
3. 点击页面中的 `选择` 按钮导入图像、模型或 TXT/CSV 文件。
4. 点击处理按钮。
5. 在右侧结果区域查看图像和曲线。
6. 点击 `保存当前结果图` 导出结果。

更详细的新手安装和使用说明请见 [INSTALL.md](INSTALL.md)。

## 模型文件

项目默认包含两个模型权重文件：

- `best_2d_denoise_model.pth`
- `peak_finder_argmax.pth`

这两个文件分别用于 AI 去噪和 AI 寻峰。如果模型文件被移动或重命名，可以在 GUI 中手动选择。

## 打包和源码可见性

本项目可以使用 PyInstaller 打包成单个 Windows exe。打包后普通用户不会直接看到 `.py` 源码文件。

需要注意的是，PyInstaller 打包不是强加密方案。对于有经验的逆向人员，仍然可能分析出部分程序逻辑。如果需要商业级源码保护，建议进一步使用代码混淆、授权校验，或将核心算法部署到服务端。

## 开发说明

推荐环境：

- Python 3.11 或更高版本
- Windows 10/11
- PyTorch
- OpenCV
- SciPy
- NumPy
- Matplotlib

## 许可证

本项目使用 MIT License。详情请见 [LICENSE](LICENSE)。
