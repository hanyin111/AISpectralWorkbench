import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable, Optional

import cv2
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import font_manager
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


APP_DIR = app_base_dir()
DEFAULT_DENOISE_MODEL = APP_DIR / "best_2d_denoise_model.pth"
DEFAULT_PEAK_MODEL = APP_DIR / "peak_finder_argmax.pth"


def configure_matplotlib_fonts() -> str:
    """Make Matplotlib use a CJK-capable font on Windows."""
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    preferred_family = "Microsoft YaHei"
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            try:
                preferred_family = font_manager.FontProperties(fname=str(font_path)).get_name()
            except Exception:
                preferred_family = "Microsoft YaHei"
            break

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        preferred_family,
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    mpl.rcParams["axes.unicode_minus"] = False
    return preferred_family


class UNet2D(nn.Module):
    def __init__(self):
        super().__init__()

        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
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
            nn.Sigmoid(),
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


class PeakFinderSoftArgmax(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, padding=14, dilation=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=15, padding=28, dilation=4),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )
        self.register_buffer("indices", torch.arange(seq_len, dtype=torch.float32))

    def forward(self, x):
        x = x.unsqueeze(1)
        logits = self.net(x).squeeze(1)
        weights = torch.softmax(logits, dim=-1)
        peak_pos = torch.sum(weights * self.indices, dim=-1)
        return peak_pos, weights


def read_gray_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise ValueError(f"无法读取图像文件: {path}")
    return image.astype(np.float32)


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return arr / (float(np.max(arr)) + 1e-8)


def spectrum_1d(img_2d: np.ndarray) -> np.ndarray:
    spec = np.sum(img_2d, axis=0)
    return spec / (float(np.max(spec)) + 1e-8)


def crop_core(img: np.ndarray, height: int = 256) -> np.ndarray:
    if img.shape[0] <= height:
        return img
    row_sum = np.sum(img, axis=1)
    brightest_row = int(np.argmax(row_sum))
    top = max(0, min(img.shape[0] - height, brightest_row - height // 2))
    return img[top : top + height, :]


def psnr(ref: np.ndarray, pred: np.ndarray) -> float:
    ref = normalize(ref)
    pred = normalize(pred)
    mse = float(np.mean((ref - pred) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def rmse(ref: np.ndarray, pred: np.ndarray) -> float:
    ref = normalize(ref)
    pred = normalize(pred)
    return float(np.sqrt(np.mean((ref - pred) ** 2)))


def denoise_dsp(noisy_img: np.ndarray, crop: bool = True) -> np.ndarray:
    work = crop_core(noisy_img) if crop else noisy_img.copy()
    denoised = median_filter(work, size=3)
    width = denoised.shape[1]
    window = min(51, width if width % 2 == 1 else width - 1)
    if window >= 7:
        denoised = savgol_filter(denoised, window_length=window, polyorder=3, axis=1)
    return np.clip(denoised, 0, None)


def denoise_ai(noisy_img: np.ndarray, model_path: str, device: torch.device) -> np.ndarray:
    model = UNet2D().to(device)
    state = torch.load(model_path, map_location=device)
    # Some demo scripts used a bare Conv2d named out_conv, while train_2d.py
    # saved the production model as Sequential(Conv2d, Sigmoid).
    if "out_conv.weight" in state:
        state["out_conv.0.weight"] = state.pop("out_conv.weight")
    if "out_conv.bias" in state:
        state["out_conv.0.bias"] = state.pop("out_conv.bias")
    model.load_state_dict(state)
    model.eval()

    norm = normalize(noisy_img)
    height, width = norm.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    padded = np.pad(norm, ((0, pad_h), (0, pad_w)), mode="edge")
    tensor_in = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        tensor_out = model(tensor_in)
    out = tensor_out.squeeze().cpu().numpy()[:height, :width]
    return np.clip(out, 0.0, 1.0)


def get_calibration_params(target_center_wl: float):
    a = (1.88082617e-08) * (target_center_wl ** 2) + (-2.14228025e-05) * target_center_wl + (1.54531419e-02)
    b = (-9.94596779e-05) * (target_center_wl ** 2) + (1.09995751e00) * target_center_wl + (-3.37890078e01)
    return a, b


def load_peak_model(model_path: str, seq_len: int, device: torch.device) -> PeakFinderSoftArgmax:
    model = PeakFinderSoftArgmax(seq_len).to(device)
    state = torch.load(model_path, map_location=device)
    if "indices" in state and tuple(state["indices"].shape) != (seq_len,):
        state = {k: v for k, v in state.items() if k != "indices"}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_txt_spectrum(path: str) -> tuple[np.ndarray, np.ndarray]:
    candidates = []
    for delimiter in (None, ",", ";", "\t"):
        for skip_header in (0, 1):
            try:
                data = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header, invalid_raise=False)
                candidates.append(data)
            except Exception:
                continue

    for data in candidates:
        if data.size == 0:
            continue
        if data.ndim == 1:
            if data.size < 2 or data.size % 2 != 0:
                continue
            data = data.reshape(-1, 2)
        if data.shape[1] < 2:
            continue
        data = data[:, :2]
        data = data[np.isfinite(data).all(axis=1)]
        if len(data):
            return data[:, 0].astype(float), data[:, 1].astype(float)
    raise ValueError("TXT 文件中没有找到有效的两列数值数据。")


def analyze_peak(image_path: str, model_path: str, center_wl: float, zolix_path: Optional[str], device: torch.device):
    img = read_gray_image(image_path)
    core = crop_core(img)
    raw_signal = np.sum(core, axis=0)
    signal_min = np.min(raw_signal)
    signal_max = np.max(raw_signal)
    ai_input_signal = (raw_signal - signal_min) / (signal_max - signal_min + 1e-8)

    model = load_peak_model(model_path, len(ai_input_signal), device)
    with torch.no_grad():
        input_tensor = torch.tensor(ai_input_signal, dtype=torch.float32).unsqueeze(0).to(device)
        predicted_pixel, _ = model(input_tensor)
        predicted_pixel = float(predicted_pixel.item())

    a, b = get_calibration_params(center_wl)
    predicted_wavelength = predicted_pixel * a + b
    x_wavelengths = np.arange(len(ai_input_signal)) * a + b

    zolix_wl = zolix_int = true_peak = error_nm = None
    if zolix_path:
        wl, raw_int = load_txt_spectrum(zolix_path)
        mask = (wl >= center_wl - 10.0) & (wl <= center_wl + 10.0)
        if np.any(mask):
            zolix_wl = wl[mask]
            local = raw_int[mask]
            zolix_int = (local - np.min(local)) / (np.max(local) - np.min(local) + 1e-8)
            true_peak = float(zolix_wl[np.argmax(zolix_int)])
            error_nm = abs(predicted_wavelength - true_peak)

    return {
        "image": img,
        "core": core,
        "signal": ai_input_signal,
        "x_wavelengths": x_wavelengths,
        "predicted_pixel": predicted_pixel,
        "predicted_wavelength": predicted_wavelength,
        "dispersion": a,
        "intercept": b,
        "zolix_wl": zolix_wl,
        "zolix_int": zolix_int,
        "true_peak": true_peak,
        "error_nm": error_nm,
    }


class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#f6f8fb")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")


class SpectralApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.plot_font = configure_matplotlib_fonts()
        self.title("AI 光谱分析工作台")
        self.geometry("1280x820")
        self.minsize(1080, 720)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_canvas = None
        self.last_figure = None
        self.last_data = None
        self._build_style()
        self._build_shell()
        self.show_home()

    def _build_style(self):
        self.configure(bg="#f6f8fb")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f6f8fb")
        style.configure("Panel.TFrame", background="#ffffff", borderwidth=1, relief="solid")
        style.configure("TLabel", background="#f6f8fb", foreground="#1b2430", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 24, "bold"), foreground="#172033")
        style.configure("Sub.TLabel", font=("Microsoft YaHei UI", 11), foreground="#5b6575")
        style.configure("Panel.TLabel", background="#ffffff", foreground="#1b2430", font=("Microsoft YaHei UI", 10))
        style.configure("PanelTitle.TLabel", background="#ffffff", foreground="#172033", font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("Metric.TLabel", background="#ffffff", foreground="#0f766e", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(12, 8))
        style.configure("Primary.TButton", background="#2563eb", foreground="#ffffff")
        style.map("Primary.TButton", background=[("active", "#1d4ed8")])
        style.configure("Card.TButton", font=("Microsoft YaHei UI", 13, "bold"), padding=(18, 18))
        style.configure("TEntry", padding=6)
        style.configure("TRadiobutton", background="#ffffff", font=("Microsoft YaHei UI", 10))
        style.configure("TCheckbutton", background="#ffffff", font=("Microsoft YaHei UI", 10))

    def _build_shell(self):
        self.header = ttk.Frame(self, padding=(28, 20, 28, 8))
        self.header.pack(fill="x")
        self.title_label = ttk.Label(self.header, text="AI 光谱分析工作台", style="Title.TLabel")
        self.title_label.pack(anchor="w")
        subtitle = f"集成 DSP/AI 去噪、AI 亚像素寻峰、Zolix TXT 可视化转换  |  当前计算设备: {self.device}"
        self.subtitle_label = ttk.Label(self.header, text=subtitle, style="Sub.TLabel")
        self.subtitle_label.pack(anchor="w", pady=(8, 0))
        self.body = ttk.Frame(self, padding=(28, 12, 28, 28))
        self.body.pack(fill="both", expand=True)

    def clear_body(self):
        for child in self.body.winfo_children():
            child.destroy()
        self.current_canvas = None
        self.last_figure = None
        self.last_data = None

    def show_home(self):
        self.clear_body()
        intro = ttk.Frame(self.body, style="Panel.TFrame", padding=24)
        intro.pack(fill="x", pady=(0, 18))
        ttk.Label(intro, text="请选择一个分析流程", style="PanelTitle.TLabel").pack(anchor="w")
        grid = ttk.Frame(self.body)
        grid.pack(fill="both", expand=True)
        for i in range(3):
            grid.columnconfigure(i, weight=1, uniform="cards")

        self._home_card(grid, 0, "光谱图像去噪", "DSP 中值滤波 + S-G 平滑，或 2D U-Net AI 去噪。支持原始图、结果图、一维光谱曲线和参考真值对比。", self.show_denoise)
        self._home_card(grid, 1, "AI 亚像素寻峰", "读取 CCD 光谱图像，利用一维卷积 Soft-Argmax 模型预测峰位，并根据中心波长动态定标为物理波长。", self.show_peak)
        self._home_card(grid, 2, "图像转换", "读取光谱仪导出的两列 TXT 数据，转换为可视化曲线并自动列出主要特征峰。", self.show_zolix)

    def _home_card(self, parent, col, title, desc, command):
        card = ttk.Frame(parent, style="Panel.TFrame", padding=22)
        card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0 if col == 2 else 10))
        ttk.Label(card, text=title, style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text=desc, style="Panel.TLabel", wraplength=310).pack(anchor="w", pady=(12, 22))
        ttk.Button(card, text="进入功能", style="Primary.TButton", command=command).pack(anchor="w")

    def _back_bar(self, title: str, desc: str):
        bar = ttk.Frame(self.body, style="Panel.TFrame", padding=18)
        bar.pack(fill="x", pady=(0, 14))
        top = ttk.Frame(bar, style="Panel.TFrame")
        top.pack(fill="x")
        ttk.Button(top, text="返回主页", command=self.show_home).pack(side="left")
        ttk.Label(top, text=title, style="PanelTitle.TLabel").pack(side="left", padx=(16, 0))
        ttk.Label(bar, text=desc, style="Panel.TLabel", wraplength=1040).pack(anchor="w", pady=(12, 0))

    def _split_layout(self):
        layout = ttk.Frame(self.body)
        layout.pack(fill="both", expand=True)
        layout.columnconfigure(0, weight=0)
        layout.columnconfigure(1, weight=1)
        layout.rowconfigure(0, weight=1)
        controls = ScrollFrame(layout)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        controls.configure(width=360)
        result = ttk.Frame(layout, style="Panel.TFrame", padding=12)
        result.grid(row=0, column=1, sticky="nsew")
        result.rowconfigure(1, weight=1)
        result.columnconfigure(0, weight=1)
        ttk.Label(result, text="结果预览", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))
        plot_holder = ttk.Frame(result, style="Panel.TFrame")
        plot_holder.grid(row=1, column=0, sticky="nsew")
        return controls.inner, plot_holder

    def _path_row(self, parent, label, variable, filetypes, save=False):
        ttk.Label(parent, text=label, style="Panel.TLabel").pack(anchor="w", pady=(12, 4))
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x")
        entry = ttk.Entry(row, textvariable=variable, width=34)
        entry.pack(side="left", fill="x", expand=True)

        def choose():
            if save:
                path = filedialog.asksaveasfilename(filetypes=filetypes)
            else:
                path = filedialog.askopenfilename(filetypes=filetypes)
            if path:
                variable.set(path)

        ttk.Button(row, text="选择", command=choose).pack(side="left", padx=(8, 0))

    def show_denoise(self):
        self.clear_body()
        self._back_bar(
            "光谱图像去噪",
            "DSP 路线使用 3x3 中值滤波抑制孤立噪点，再沿色散方向做 Savitzky-Golay 平滑以保持峰形；AI 路线使用二维 U-Net 重建低信噪比光谱图像。",
        )
        controls, plot_holder = self._split_layout()
        mode = tk.StringVar(value="DSP")
        noisy = tk.StringVar()
        reference = tk.StringVar()
        model_path = tk.StringVar(value=str(DEFAULT_DENOISE_MODEL if DEFAULT_DENOISE_MODEL.exists() else ""))
        crop = tk.BooleanVar(value=True)

        ttk.Label(controls, text="去噪方式", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Radiobutton(controls, text="DSP 级联滤波", value="DSP", variable=mode).pack(anchor="w", pady=2)
        ttk.Radiobutton(controls, text="AI 2D U-Net", value="AI", variable=mode).pack(anchor="w", pady=2)
        self._path_row(controls, "待去噪光谱图像 (.tif/.png/.jpg)", noisy, [("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        self._path_row(controls, "参考真值图像，可选", reference, [("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        self._path_row(controls, "AI 去噪模型，可选", model_path, [("PyTorch model", "*.pth *.pt"), ("All files", "*.*")])
        ttk.Checkbutton(controls, text="DSP 自动裁剪 256 像素核心光谱区", variable=crop).pack(anchor="w", pady=(12, 0))
        ttk.Button(
            controls,
            text="开始去噪并显示结果",
            style="Primary.TButton",
            command=lambda: self._run_async(lambda: self._process_denoise(mode.get(), noisy.get(), reference.get(), model_path.get(), crop.get(), plot_holder)),
        ).pack(fill="x", pady=(22, 10))
        ttk.Button(controls, text="保存当前结果图", command=self._save_figure).pack(fill="x")

    def _process_denoise(self, mode, noisy_path, ref_path, model_path, crop, plot_holder):
        if not noisy_path:
            raise ValueError("请先选择待去噪的光谱图像。")
        noisy_img = read_gray_image(noisy_path)
        ref_img = read_gray_image(ref_path) if ref_path else None
        if mode == "AI":
            if not model_path:
                raise ValueError("AI 去噪需要选择 .pth 模型文件。")
            denoised = denoise_ai(noisy_img, model_path, self.device)
            display_noisy = normalize(noisy_img)
            title = "AI 2D U-Net 去噪结果"
        else:
            denoised = denoise_dsp(noisy_img, crop=crop)
            display_noisy = normalize(crop_core(noisy_img) if crop else noisy_img)
            if ref_img is not None and crop:
                ref_img = crop_core(ref_img)
            title = "DSP 中值滤波 + Savitzky-Golay 去噪结果"

        fig = Figure(figsize=(9.4, 6.5), dpi=100)
        fig.suptitle(title, fontsize=14, fontweight="bold")
        cols = 3 if ref_img is not None else 2
        axes = fig.subplots(2, cols)
        axes = np.atleast_2d(axes)
        panels = [("原始图像", display_noisy), ("去噪结果", normalize(denoised))]
        if ref_img is not None:
            panels.append(("参考真值", normalize(ref_img)))
        for i, (name, img) in enumerate(panels):
            axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[0, i].set_title(name)
            axes[0, i].axis("off")
            axes[1, i].plot(spectrum_1d(img), linewidth=1.6)
            axes[1, i].set_title(f"{name} 一维光谱")
            axes[1, i].grid(True, linestyle="--", alpha=0.35)
        if ref_img is not None:
            fig.text(0.02, 0.02, f"RMSE: {rmse(ref_img, denoised):.4f}    PSNR: {psnr(ref_img, denoised):.2f} dB", fontsize=10)
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        self._show_figure(plot_holder, fig)

    def show_peak(self):
        self.clear_body()
        self._back_bar(
            "AI 亚像素寻峰",
            "先从 CCD 图像中自动裁剪最亮的 256 像素光谱带，压缩为一维信号；随后使用一维卷积 Soft-Argmax 模型输出亚像素峰位，并通过中心波长相关的动态多项式定标换算为 nm。",
        )
        controls, plot_holder = self._split_layout()
        image_path = tk.StringVar()
        zolix_path = tk.StringVar()
        model_path = tk.StringVar(value=str(DEFAULT_PEAK_MODEL if DEFAULT_PEAK_MODEL.exists() else ""))
        center_wl = tk.StringVar(value="541.0")

        self._path_row(controls, "CCD 光谱图像 (.tif/.png/.jpg)", image_path, [("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        self._path_row(controls, "Zolix 慢扫描 TXT，可选", zolix_path, [("Text files", "*.txt *.csv"), ("All files", "*.*")])
        self._path_row(controls, "AI 寻峰模型 (.pth)", model_path, [("PyTorch model", "*.pth *.pt"), ("All files", "*.*")])
        ttk.Label(controls, text="光谱仪中心波长 (nm)", style="Panel.TLabel").pack(anchor="w", pady=(12, 4))
        ttk.Entry(controls, textvariable=center_wl).pack(fill="x")
        ttk.Button(
            controls,
            text="开始寻峰并显示结果",
            style="Primary.TButton",
            command=lambda: self._run_async(lambda: self._process_peak(image_path.get(), model_path.get(), center_wl.get(), zolix_path.get(), plot_holder)),
        ).pack(fill="x", pady=(22, 10))
        ttk.Button(controls, text="保存当前结果图", command=self._save_figure).pack(fill="x")

    def _process_peak(self, image_path, model_path, center_wl, zolix_path, plot_holder):
        if not image_path:
            raise ValueError("请先选择 CCD 光谱图像。")
        if not model_path:
            raise ValueError("请先选择 AI 寻峰模型。")
        result = analyze_peak(image_path, model_path, float(center_wl), zolix_path or None, self.device)
        fig = Figure(figsize=(9.4, 6.5), dpi=100)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25])
        ax_img = fig.add_subplot(gs[0, 0])
        ax_plot = fig.add_subplot(gs[1, 0])
        ax_img.imshow(normalize(result["core"]), cmap="gray", vmin=0, vmax=1)
        ax_img.set_title("CCD 核心光谱区")
        ax_img.axis("off")

        ax_plot.plot(result["x_wavelengths"], result["signal"], color="#64748b", linewidth=2.2, alpha=0.75, label="CCD 瞬态信号")
        if result["zolix_wl"] is not None:
            ax_plot.plot(result["zolix_wl"], result["zolix_int"], color="#2563eb", linewidth=1.8, label="Zolix 慢扫描真值")
            ax_plot.axvline(result["true_peak"], color="#06b6d4", linestyle=":", linewidth=2, label=f"Zolix 峰位: {result['true_peak']:.3f} nm")
        ax_plot.axvline(result["predicted_wavelength"], color="#dc2626", linestyle="--", linewidth=2, label=f"AI 峰位: {result['predicted_wavelength']:.3f} nm")
        ax_plot.set_xlabel("绝对波长 (nm)")
        ax_plot.set_ylabel("归一化强度")
        ax_plot.grid(True, linestyle="--", alpha=0.35)
        ax_plot.legend(loc="upper right", fontsize=9)
        ax_plot.set_xlim(result["predicted_wavelength"] - 5, result["predicted_wavelength"] + 5)
        report = [
            f"AI 亚像素坐标: {result['predicted_pixel']:.3f} px",
            f"色散系数 A: {result['dispersion']:.5f} nm/px",
            f"AI 波长: {result['predicted_wavelength']:.3f} nm",
        ]
        if result["error_nm"] is not None:
            report.append(f"绝对误差: {result['error_nm']:.4f} nm")
        ax_plot.text(0.02, 0.95, "\n".join(report), transform=ax_plot.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.88, edgecolor="#cbd5e1"))
        fig.tight_layout()
        self._show_figure(plot_holder, fig)

    def show_zolix(self):
        self.clear_body()
        self._back_bar(
            "Zolix TXT 可视化转换",
            "读取 Zolix 光谱仪导出的两列文本数据，将波长-强度表格转换为曲线视图，并基于 prominence 自动提取主要特征峰。",
        )
        controls, plot_holder = self._split_layout()
        txt_path = tk.StringVar()
        prominence = tk.StringVar(value="10")
        self._path_row(controls, "Zolix TXT/CSV 文件", txt_path, [("Text files", "*.txt *.csv"), ("All files", "*.*")])
        ttk.Label(controls, text="寻峰阈值 prominence (% 最大强度)", style="Panel.TLabel").pack(anchor="w", pady=(12, 4))
        ttk.Entry(controls, textvariable=prominence).pack(fill="x")
        ttk.Button(
            controls,
            text="转换并显示曲线",
            style="Primary.TButton",
            command=lambda: self._run_async(lambda: self._process_zolix(txt_path.get(), prominence.get(), plot_holder)),
        ).pack(fill="x", pady=(22, 10))
        ttk.Button(controls, text="保存当前结果图", command=self._save_figure).pack(fill="x")

    def _process_zolix(self, txt_path, prominence_pct, plot_holder):
        if not txt_path:
            raise ValueError("请先选择 Zolix TXT 文件。")
        wl, intensity = load_txt_spectrum(txt_path)
        prom = np.max(intensity) * max(0.0, float(prominence_pct)) / 100.0
        peaks, _props = find_peaks(intensity, prominence=prom)
        order = np.argsort(intensity[peaks])[::-1] if len(peaks) else []
        top = peaks[order[:8]] if len(peaks) else []

        fig = Figure(figsize=(9.4, 6.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(wl, intensity, color="#2563eb", linewidth=1.6, label="Zolix 原始光谱")
        if len(top):
            ax.scatter(wl[top], intensity[top], color="#dc2626", s=36, zorder=4, label="主要特征峰")
            for idx in top[:6]:
                ax.annotate(f"{wl[idx]:.2f} nm", (wl[idx], intensity[idx]), textcoords="offset points", xytext=(5, 8), fontsize=9)
        ax.set_title("Zolix TXT 光谱曲线")
        ax.set_xlabel("波长 (nm)")
        ax.set_ylabel("强度")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        if len(top):
            rows = [f"{i + 1}. {wl[idx]:.2f} nm  |  {intensity[idx]:.3g}" for i, idx in enumerate(top[:6])]
            ax.text(0.02, 0.96, "最强峰:\n" + "\n".join(rows), transform=ax.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.88, edgecolor="#cbd5e1"))
        fig.tight_layout()
        self._show_figure(plot_holder, fig)

    def _show_figure(self, holder, fig):
        def draw():
            for child in holder.winfo_children():
                child.destroy()
            self.last_figure = fig
            self.current_canvas = FigureCanvasTkAgg(fig, master=holder)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.after(0, draw)

    def _run_async(self, task: Callable[[], None]):
        def runner():
            try:
                task()
            except Exception as exc:
                traceback.print_exc()
                error_text = str(exc)
                self.after(0, lambda: messagebox.showerror("处理失败", error_text))

        threading.Thread(target=runner, daemon=True).start()

    def _save_figure(self):
        if self.last_figure is None:
            messagebox.showinfo("暂无结果", "请先运行一次处理流程。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")])
        if path:
            self.last_figure.savefig(path, dpi=180, bbox_inches="tight")
            messagebox.showinfo("保存成功", f"结果图已保存到:\n{path}")


if __name__ == "__main__":
    try:
        app = SpectralApp()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("启动失败", f"{e}\n\n{traceback.format_exc()}")
        sys.exit(1)
