# AI Spectral Analysis Workbench

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)](#)

中文说明请见 [README_CN.md](README_CN.md).

AI Spectral Analysis Workbench is a Windows desktop application for photoluminescence spectral image processing. It integrates traditional DSP denoising, AI-based 2D U-Net denoising, AI sub-pixel peak localization, and Zolix TXT/CSV spectrum visualization into one mouse-driven graphical interface.

The project was built for weak-light spectral analysis workflows where CCD spectral images and Zolix slow-scan data need to be compared, visualized, and processed quickly.

## Features

- **Spectral image denoising**
  - Traditional DSP mode: median filtering plus Savitzky-Golay smoothing.
  - AI mode: 2D U-Net denoising with `best_2d_denoise_model.pth`.
  - Displays 2D image comparison and 1D spectrum curves.
  - Supports optional reference image comparison with RMSE and PSNR.

- **AI sub-pixel peak localization**
  - Loads a CCD spectral image.
  - Crops the brightest spectral band automatically.
  - Uses `peak_finder_argmax.pth` to predict the sub-pixel peak position.
  - Converts pixel position to physical wavelength with a dynamic calibration model.
  - Optionally compares against Zolix slow-scan TXT/CSV data.

- **Zolix TXT/CSV visualization**
  - Reads two-column wavelength-intensity files.
  - Plots spectrum curves.
  - Detects and labels major peaks automatically.

- **Desktop GUI**
  - Built with Tkinter and Matplotlib.
  - All major operations can be completed by mouse clicks.
  - Result figures can be exported as PNG or PDF.

## Repository Contents

| Path | Description |
| --- | --- |
| `spectral_analysis_gui.py` | Main GUI application |
| `run_gui.bat` | Starts the application from source on Windows |
| `build_exe.bat` | Builds a standalone Windows exe with PyInstaller |
| `requirements.txt` | Runtime Python dependencies |
| `requirements-dev.txt` | Runtime dependencies plus packaging tools |
| `best_2d_denoise_model.pth` | 2D U-Net denoising model weights |
| `peak_finder_argmax.pth` | AI peak localization model weights |
| `traditional.py` | Original DSP denoising demo script |
| `test_2d_model.py` | Original AI denoising demo script |
| `model.py` | Original AI peak localization demo script |
| `zolixtrans.py` | Original Zolix TXT peak extraction demo script |
| `INSTALL.md` | Detailed installation guide for beginners |
| `LICENSE` | MIT license |

## Quick Start

### Option 1: Run from source

Install Python 3.11 or newer, then run:

```bat
python -m pip install -r requirements.txt
python spectral_analysis_gui.py
```

On Windows, you can also double-click:

```text
run_gui.bat
```

### Option 2: Build a standalone exe

Install development dependencies:

```bat
python -m pip install -r requirements-dev.txt
```

Build the executable:

```bat
python -m PyInstaller --noconfirm --clean --windowed --onefile --name AISpectralWorkbench --add-data "best_2d_denoise_model.pth;." --add-data "peak_finder_argmax.pth;." spectral_analysis_gui.py
```

Or double-click:

```text
build_exe.bat
```

The generated exe will be created at:

```text
dist\AISpectralWorkbench.exe
```

> Note: The packaged exe is large because it includes PyTorch, OpenCV, Matplotlib, and model weights.

## Input Data Formats

### Spectral images

Supported image formats:

- `.tif`
- `.tiff`
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`

For best results, use the original CCD spectral image files from the experiment.

### Zolix TXT/CSV files

The Zolix data file should contain at least two numeric columns:

```text
wavelength intensity
300.0 120
300.1 125
300.2 130
```

Comma-separated files are also supported:

```text
wavelength,intensity
300.0,120
300.1,125
300.2,130
```

## Usage

1. Start the application.
2. Choose one of the three main functions:
   - `光谱图像去噪`
   - `AI 亚像素寻峰`
   - `图像转换`
3. Select the required image, model, or TXT/CSV files with the `选择` buttons.
4. Click the processing button.
5. Review the generated plots in the result panel.
6. Click `保存当前结果图` to export the current figure.

For a step-by-step guide written for complete beginners, see [INSTALL.md](INSTALL.md).

## Model Files

This repository includes two model weight files:

- `best_2d_denoise_model.pth`
- `peak_finder_argmax.pth`

They are required for the default AI denoising and AI peak localization workflows. If they are moved or renamed, select the correct model manually in the GUI.

## Packaging and Source Visibility

The project can be packaged into a single Windows exe with PyInstaller. The generated exe does not directly expose `.py` source files to ordinary users.

However, PyInstaller packaging is not strong encryption. A determined reverse engineer may still recover parts of the program logic. For commercial-grade protection, consider code obfuscation, license verification, or moving core algorithms to a server-side API.

## Development Notes

- Python version: 3.11 or newer is recommended.
- Main GUI framework: Tkinter.
- Plotting: Matplotlib.
- AI inference: PyTorch.
- Image processing: OpenCV, SciPy, NumPy.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
