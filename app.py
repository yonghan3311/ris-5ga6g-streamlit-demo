"""Interactive Streamlit demo for the RIS-assisted mmWave simulation."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ris_simulator import BS, BUILDINGS, RIS, RisParams, run_scenarios, sample_users  # noqa: E402


PALETTE_RSRP = (
    (0.00, (30, 44, 120)),
    (0.22, (35, 91, 168)),
    (0.48, (28, 155, 165)),
    (0.74, (122, 199, 82)),
    (1.00, (248, 221, 61)),
)
PALETTE_GAIN = (
    (0.00, (250, 252, 255)),
    (0.25, (203, 225, 247)),
    (0.55, (86, 157, 211)),
    (0.80, (55, 86, 168)),
    (1.00, (75, 31, 122)),
)
PALETTE_SINR = (
    (0.00, (58, 38, 115)),
    (0.24, (47, 85, 151)),
    (0.50, (33, 145, 140)),
    (0.74, (98, 190, 99)),
    (1.00, (236, 226, 75)),
)


def font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def lerp_color(t: float, stops: Sequence[Tuple[float, Tuple[int, int, int]]]) -> Tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        if t <= p1:
            u = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            return tuple(int(c0[j] + u * (c1[j] - c0[j])) for j in range(3))
    return stops[-1][1]


def world_to_pixel(x: float, y: float, area, span: float):
    x0, y0, x1, y1 = area
    return (
        int(x0 + (x + span) / (2 * span) * (x1 - x0)),
        int(y1 - (y + span) / (2 * span) * (y1 - y0)),
    )


def heatmap_image(title: str, axis: np.ndarray, data: np.ndarray, params: RisParams, vmin: float, vmax: float, unit: str, palette) -> Image.Image:
    width, height = 980, 650
    area = (70, 72, 775, 565)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((24, 22), title, font=font(22, True), fill=(17, 24, 39))
    draw.line((24, 56, width - 24, 56), fill=(214, 222, 232), width=2)

    clipped = np.clip((data - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgb = np.zeros((clipped.shape[0], clipped.shape[1], 3), dtype=np.uint8)
    for r in range(clipped.shape[0]):
        for c in range(clipped.shape[1]):
            rgb[r, c] = lerp_color(float(clipped[r, c]), palette)
    hm = Image.fromarray(rgb, mode="RGB").resize((area[2] - area[0], area[3] - area[1]), Image.Resampling.BICUBIC)
    img.paste(hm, area[:2])
    draw.rectangle(area, outline=(34, 34, 34), width=2)

    span = float(axis[-1])
    for tick in np.linspace(-span, span, 5):
        px, _ = world_to_pixel(float(tick), -span, area, span)
        _, py = world_to_pixel(-span, float(tick), area, span)
        draw.line((px, area[3], px, area[3] + 5), fill=(34, 34, 34), width=1)
        draw.line((area[0] - 5, py, area[0], py), fill=(34, 34, 34), width=1)
        draw.text((px - 18, area[3] + 12), f"{tick:.0f}", font=font(12), fill=(34, 34, 34))
        draw.text((area[0] - 48, py - 8), f"{tick:.0f}", font=font(12), fill=(34, 34, 34))
    draw.text(((area[0] + area[2]) // 2 - 20, 612), "x / m", font=font(14), fill=(34, 34, 34))
    draw.text((18, 72), "y / m", font=font(14), fill=(34, 34, 34))

    for x0, y0, x1, y1 in BUILDINGS:
        p0 = world_to_pixel(x0, y0, area, span)
        p1 = world_to_pixel(x1, y1, area, span)
        draw.rectangle((p0[0], p1[1], p1[0], p0[1]), fill=(34, 42, 54), outline=(17, 24, 39))
    ris = np.array([params.ris_x_m, params.ris_y_m])
    for label, point, color in [("BS", BS, (239, 68, 68)), ("RIS", ris, (34, 197, 94))]:
        px, py = world_to_pixel(float(point[0]), float(point[1]), area, span)
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color, outline=(255, 255, 255), width=2)
        draw.text((px + 10, py - 18), label, font=font(15, True), fill=color)

    bar = (825, 94, 858, 535)
    for i in range(bar[3] - bar[1]):
        t = 1 - i / max(bar[3] - bar[1] - 1, 1)
        draw.line((bar[0], bar[1] + i, bar[2], bar[1] + i), fill=lerp_color(t, palette))
    draw.rectangle(bar, outline=(17, 24, 39), width=1)
    for i, value in enumerate(np.linspace(vmax, vmin, 6)):
        y = bar[1] + i * (bar[3] - bar[1]) / 5
        draw.text((bar[2] + 10, y - 8), f"{value:.0f} {unit}", font=font(12), fill=(34, 34, 34))
    return img


@st.cache_data(show_spinner=False)
def cached_run(params_tuple):
    params = RisParams(*params_tuple)
    return run_scenarios(params)


def main() -> None:
    st.set_page_config(page_title="RIS辅助毫米波覆盖仿真", layout="wide")
    st.title("面向5G-A/6G的RIS智能反射面辅助毫米波覆盖增强仿真")
    st.caption("左侧调节参数，右侧实时运行仿真并展示覆盖、干扰和链路结果。")

    with st.sidebar:
        st.header("仿真参数")
        carrier = st.slider("载波频率/GHz", 3.5, 60.0, 28.0, 0.5)
        bandwidth = st.slider("系统带宽/MHz", 50.0, 800.0, 400.0, 50.0)
        power = st.slider("发射功率/dBm", 20.0, 42.0, 30.0, 1.0)
        elements = st.select_slider("RIS单元数量", options=[16, 32, 64, 128, 256, 512, 1024], value=256)
        blockage = st.slider("遮挡损耗/dB", 8.0, 36.0, 24.0, 2.0)
        ris_x = st.slider("RIS横坐标x/m", 60.0, 260.0, 135.0, 5.0)
        ris_y = st.slider("RIS纵坐标y/m", 20.0, 240.0, 95.0, 5.0)
        view = st.radio("结果视图", ["无RIS RSRP", "优化RIS RSRP", "RIS RSRP增益", "优化RIS SINR", "优化RIS吞吐率"])

    params = RisParams(
        carrier_ghz=carrier,
        bandwidth_mhz=bandwidth,
        tx_power_dbm=power,
        ris_elements=int(elements),
        blockage_loss_db=blockage,
        grid_points=101,
        ris_x_m=ris_x,
        ris_y_m=ris_y,
    )
    data = run_scenarios(params)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("无RIS覆盖率", f"{data['coverage_no'] * 100:.2f}%")
    c2.metric("优化RIS覆盖率", f"{data['coverage_opt'] * 100:.2f}%", f"{(data['coverage_opt'] - data['coverage_no']) * 100:.2f} pct")
    c3.metric("遮挡区平均速率", f"{data['blocked_mean_rate_opt']:.2f} Mbps")
    c4.metric("弱覆盖区平均增益", f"{data['weak_region_mean_gain']:.2f} dB")

    if view == "无RIS RSRP":
        image = heatmap_image("No-RIS RSRP Heatmap", data["x"], data["no_ris"], params, -125, -62, "dBm", PALETTE_RSRP)
    elif view == "优化RIS RSRP":
        image = heatmap_image("Optimized RIS RSRP Heatmap", data["x"], data["with_opt"], params, -125, -62, "dBm", PALETTE_RSRP)
    elif view == "RIS RSRP增益":
        image = heatmap_image("RSRP Gain with RIS", data["x"], data["with_opt"] - data["no_ris"], params, 0, 28, "dB", PALETTE_GAIN)
    elif view == "优化RIS SINR":
        image = heatmap_image("Optimized RIS SINR Map", data["x"], data["sinr_opt"], params, -10, 42, "dB", PALETTE_SINR)
    else:
        image = heatmap_image("Optimized RIS Throughput Map", data["x"], data["throughput_opt"], params, 0, 4200, "Mbps", PALETTE_SINR)
    st.image(image, use_container_width=True)

    st.subheader("典型用户链路质量")
    st.dataframe(sample_users(data), use_container_width=True)

    st.subheader("仿真过程说明")
    st.markdown(
        """
        1. 建立二维城区网格，并设置基站、RIS、建筑物遮挡和用户位置。  
        2. 判断每个网格点的BS-UE与RIS-UE链路是否被遮挡。  
        3. 用Close-in路径损耗模型计算直接链路和RIS级联反射链路。  
        4. 在毫瓦域合并接收功率，加入同频干扰和热噪声，计算SINR与吞吐率。  
        5. 对比无RIS、随机相位RIS和优化相位RIS的覆盖增强效果。  
        """
    )


if __name__ == "__main__":
    main()
