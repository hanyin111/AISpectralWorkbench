@echo off
cd /d "%~dp0"
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --windowed --onefile --name AISpectralWorkbench --add-data "best_2d_denoise_model.pth;." --add-data "peak_finder_argmax.pth;." spectral_analysis_gui.py
pause
