# AI 光谱分析工作台安装说明

本文档假设使用者完全没有安装过 Python，也不熟悉命令行。请按顺序操作。

## 一、最简单方式：直接运行 exe

如果你已经拿到了打包好的 `AISpectralWorkbench.exe`，这是推荐方式。

### 1. 准备文件夹

建议新建一个文件夹，例如：

```text
D:\SpectrometerAI
```

把下面文件放进去：

```text
AISpectralWorkbench.exe
```

如果打包方式是单文件 exe，通常只需要这一个文件即可。

### 2. 启动程序

双击：

```text
AISpectralWorkbench.exe
```

等待几秒到几十秒。第一次启动可能稍慢，因为程序会解压内部运行文件。

### 3. Windows 安全提示

如果 Windows 提示“Windows 已保护你的电脑”：

1. 点击 `更多信息`。
2. 点击 `仍要运行`。

这是因为 exe 没有购买数字签名证书，并不一定表示程序有问题。

### 4. 使用程序

程序打开后，主页有三个功能：

1. `光谱图像去噪`
2. `AI 亚像素寻峰`
3. `图像转换`

所有文件都通过 `选择` 按钮导入。

## 二、源码方式运行

如果没有 exe，或者需要修改程序，就使用源码方式。

## 1. 安装 Python

### 1.1 下载 Python

打开浏览器，访问：

```text
https://www.python.org/downloads/
```

点击下载 Python。推荐安装：

```text
Python 3.11.x
```

### 1.2 安装 Python

双击下载好的安装包。

在第一个安装界面，请一定勾选：

```text
Add python.exe to PATH
```

然后点击：

```text
Install Now
```

等待安装完成。

### 1.3 检查 Python 是否安装成功

按键盘：

```text
Win + R
```

输入：

```text
cmd
```

点击确定。

在黑色窗口中输入：

```bat
python --version
```

如果看到类似下面内容，说明安装成功：

```text
Python 3.11.9
```

如果提示“不是内部或外部命令”，通常是安装时没有勾选 `Add python.exe to PATH`，请重新安装 Python 并勾选它。

## 2. 准备程序文件

建议新建文件夹：

```text
D:\SpectrometerAI
```

把本项目文件复制进去，至少需要：

```text
spectral_analysis_gui.py
run_gui.bat
best_2d_denoise_model.pth
peak_finder_argmax.pth
```

为了保留原始演示代码，也可以一起保留：

```text
traditional.py
test_2d_model.py
model.py
zolixtrans.py
```

## 3. 安装依赖

### 3.1 打开命令行

进入程序文件夹。

如果文件夹是：

```text
D:\SpectrometerAI
```

可以在命令行输入：

```bat
D:
cd D:\SpectrometerAI
```

### 3.2 升级 pip

输入：

```bat
python -m pip install --upgrade pip
```

### 3.3 安装基础依赖

输入：

```bat
python -m pip install numpy scipy matplotlib opencv-python pillow python-docx
```

### 3.4 安装 PyTorch

如果电脑没有 Nvidia 显卡，或者不确定有没有显卡，建议安装 CPU 版本：

```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

如果电脑有 Nvidia 显卡，并且已经正确安装 CUDA，可以去 PyTorch 官网选择对应命令：

```text
https://pytorch.org/get-started/locally/
```

不会配置 CUDA 的用户建议直接使用 CPU 版本，程序可以正常运行，只是 AI 推理速度可能慢一些。

## 4. 启动程序

方式一：双击：

```text
run_gui.bat
```

方式二：命令行运行：

```bat
python spectral_analysis_gui.py
```

## 5. 光谱图像去噪功能使用

### 5.1 DSP 去噪

1. 点击主页 `光谱图像去噪`。
2. 选择 `DSP 级联滤波`。
3. 点击 `待去噪光谱图像` 后面的 `选择`。
4. 选择 `.tif`、`.png`、`.jpg` 等光谱图像。
5. 如有参考真值图像，可在 `参考真值图像，可选` 处选择。
6. 点击 `开始去噪并显示结果`。

程序会显示：

- 原始图像
- 去噪结果
- 一维光谱曲线
- 如果选择了参考真值，会显示 RMSE 和 PSNR

### 5.2 AI 去噪

1. 点击主页 `光谱图像去噪`。
2. 选择 `AI 2D U-Net`。
3. 选择待去噪光谱图像。
4. `AI 去噪模型` 默认会指向 `best_2d_denoise_model.pth`。
5. 如果默认路径为空，手动选择 `best_2d_denoise_model.pth`。
6. 点击 `开始去噪并显示结果`。

注意：AI 去噪需要 PyTorch。如果报错提示 `No module named torch`，说明还没有安装 PyTorch，请回到本文档第 3.4 步。

## 6. AI 亚像素寻峰功能使用

1. 点击主页 `AI 亚像素寻峰`。
2. 在 `CCD 光谱图像` 处选择实验采集的 CCD 光谱图像。
3. 在 `Zolix 慢扫描 TXT，可选` 处选择 Zolix 导出的 TXT 文件。
4. 在 `AI 寻峰模型` 处选择 `peak_finder_argmax.pth`。
5. 在 `光谱仪中心波长 (nm)` 中输入实验时设置的中心波长，例如：

```text
541.0
```

6. 点击 `开始寻峰并显示结果`。

程序会输出：

- CCD 核心光谱区
- CCD 瞬态信号
- AI 预测峰位
- 可选的 Zolix 慢扫描真值
- AI 波长、亚像素坐标、绝对误差

## 7. 图像转换功能使用

这里的“图像转换”指的是把 Zolix 光谱仪导出的 TXT/CSV 数据转换成可视化曲线图。

1. 点击主页 `图像转换`。
2. 在 `Zolix TXT/CSV 文件` 处点击 `选择`。
3. 选择 `.txt` 或 `.csv` 文件。
4. `寻峰阈值 prominence` 默认是 `10`，表示峰值显著性阈值为最大强度的 10%。
5. 点击 `转换并显示曲线`。

程序会绘制光谱曲线，并自动标注主要特征峰。

## 8. 保存结果图

每个功能页面都有：

```text
保存当前结果图
```

点击后可以选择保存为：

- PNG 图片
- PDF 文件

## 9. 常见问题

### 9.1 双击 run_gui.bat 后窗口一闪而过

可能原因：

- Python 没装好。
- 依赖没安装。
- 程序文件缺失。

解决方法：

1. 在程序文件夹空白处按住 `Shift`，点击鼠标右键。
2. 选择 `在终端中打开` 或 `在此处打开 PowerShell 窗口`。
3. 输入：

```bat
python spectral_analysis_gui.py
```

查看报错内容。

### 9.2 提示 No module named xxx

表示缺少依赖。例如：

```text
No module named cv2
```

说明没有安装 OpenCV。可以重新执行：

```bat
python -m pip install numpy scipy matplotlib opencv-python pillow python-docx
```

如果是：

```text
No module named torch
```

执行：

```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 9.3 AI 去噪模型加载失败

请检查：

1. 是否选择了正确的 `best_2d_denoise_model.pth`。
2. 模型文件是否完整。
3. 程序是否使用当前版本的 `spectral_analysis_gui.py`。

### 9.4 AI 寻峰模型加载失败

请检查：

1. 是否选择了 `peak_finder_argmax.pth`。
2. CCD 图像宽度是否与模型训练时一致。当前模型主要面向 2048 像素宽度的 CCD 数据。

### 9.5 中文显示为方块

程序已内置 Windows 中文字体配置，优先使用微软雅黑。如果仍然显示异常，请确认系统中存在：

```text
C:\Windows\Fonts\msyh.ttc
```

或：

```text
C:\Windows\Fonts\simhei.ttf
```

### 9.6 TXT 文件无法读取

TXT/CSV 至少需要两列数值：

```text
波长 强度
```

如果文件开头有大量说明文字，请删除说明文字，只保留表头和数据。

### 9.7 程序运行很慢

常见原因：

- 使用 CPU 运行 AI 模型。
- 输入图像很大。
- 第一次打开 exe 时需要解压运行环境。

可以尝试：

1. 等待程序处理完成。
2. 使用较小的测试图像。
3. 如果电脑有 Nvidia 显卡，再配置 CUDA 版本 PyTorch。

## 10. 重新打包 exe

如果需要自己从源码重新生成 exe，请先安装 PyInstaller：

```bat
python -m pip install pyinstaller
```

然后在程序目录运行：

```bat
pyinstaller --noconfirm --clean --windowed --onefile --name AISpectralWorkbench --add-data "best_2d_denoise_model.pth;." --add-data "peak_finder_argmax.pth;." spectral_analysis_gui.py
```

打包完成后，exe 会出现在：

```text
dist\AISpectralWorkbench.exe
```

注意：因为程序包含 PyTorch、OpenCV 和 Matplotlib，生成的 exe 文件可能比较大，打包过程也可能需要几分钟。

## 11. 源码保护说明

PyInstaller 打包后，普通用户不会看到 `.py` 源码文件，只会看到 exe。但它不是绝对加密方案，专业逆向仍有可能分析程序逻辑。

如果需要更强保护，可以考虑：

1. 使用代码混淆工具。
2. 将核心算法部署到服务器，客户端只负责上传数据和显示结果。
3. 增加授权码或设备绑定机制。
4. 使用商业级加壳或授权系统。

