"""Simulation core for a 5G-A/6G RIS-assisted mmWave cellular lab project.

The models are intentionally compact and deterministic so the results can be
explained in an undergraduate mobile communications report. Distances are in
meters unless otherwise noted.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, log10, pi, sqrt
from typing import Dict, Iterable, List, Tuple

import numpy as np


C = 299_792_458.0


@dataclass(frozen=True)
class RisParams:
    carrier_ghz: float = 28.0
    bandwidth_mhz: float = 400.0
    tx_power_dbm: float = 30.0
    bs_gain_dbi: float = 24.0
    ue_gain_dbi: float = 3.0
    noise_figure_db: float = 7.0
    ris_elements: int = 256
    ris_reflection_loss_db: float = 1.5
    ris_phase_efficiency_db: float = 8.0
    n_los: float = 2.05
    n_nlos: float = 3.65
    blockage_loss_db: float = 24.0
    shadow_std_db: float = 3.0
    grid_span_m: float = 420.0
    grid_points: int = 121
    ris_x_m: float = 135.0
    ris_y_m: float = 95.0


BS = np.array([0.0, 0.0])
RIS = np.array([135.0, 95.0])
BUILDINGS = [
    (-65.0, 60.0, 70.0, 185.0),
    (90.0, -215.0, 168.0, -35.0),
]


def ris_position(params: RisParams) -> np.ndarray:
    return np.array([params.ris_x_m, params.ris_y_m], dtype=float)
INTERFERERS = [
    np.array([520.0, 0.0]),
    np.array([-520.0, 0.0]),
    np.array([260.0, 450.0]),
    np.array([-260.0, 450.0]),
    np.array([260.0, -450.0]),
    np.array([-260.0, -450.0]),
]


def db_to_linear(db: np.ndarray | float) -> np.ndarray | float:
    return np.power(10.0, np.asarray(db) / 10.0)


def linear_to_db(value: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(np.asarray(value), 1e-30))


def noise_power_dbm(params: RisParams) -> float:
    return -174.0 + 10.0 * log10(params.bandwidth_mhz * 1e6) + params.noise_figure_db


def fspl_1m_db(carrier_ghz: float) -> float:
    wavelength = C / (carrier_ghz * 1e9)
    return 20.0 * log10(4.0 * pi / wavelength)


def ci_path_loss_db(
    distance_m: np.ndarray | float,
    params: RisParams,
    los: np.ndarray | bool = True,
    extra_blockage_db: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Close-in 1 m reference path loss with deterministic shadowing."""
    d = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    los_arr = np.asarray(los)
    exponent = np.where(los_arr, params.n_los, params.n_nlos)
    shadow = params.shadow_std_db * np.sin(0.013 * d + 0.7 * np.log1p(d))
    return fspl_1m_db(params.carrier_ghz) + 10.0 * exponent * np.log10(d) + shadow + extra_blockage_db


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def line_blocked_scalar(a: np.ndarray, b: np.ndarray, buildings: Iterable[Tuple[float, float, float, float]] = BUILDINGS) -> bool:
    for x0, y0, x1, y1 in buildings:
        corners = [
            np.array([x0, y0]),
            np.array([x1, y0]),
            np.array([x1, y1]),
            np.array([x0, y1]),
        ]
        edges = zip(corners, corners[1:] + corners[:1])
        if any(_segments_intersect(a, b, c, d) for c, d in edges):
            return True
    return False


def blocked_mask(source: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    mask = np.fromiter((line_blocked_scalar(source, point) for point in flat), dtype=bool, count=flat.shape[0])
    return mask.reshape(xx.shape)


def received_direct_dbm(source: np.ndarray, xx: np.ndarray, yy: np.ndarray, params: RisParams, tx_power_dbm: float | None = None) -> np.ndarray:
    tx = params.tx_power_dbm if tx_power_dbm is None else tx_power_dbm
    d = np.sqrt((xx - source[0]) ** 2 + (yy - source[1]) ** 2)
    blocked = blocked_mask(source, xx, yy)
    pl = ci_path_loss_db(d, params, los=~blocked, extra_blockage_db=np.where(blocked, params.blockage_loss_db, 0.0))
    return tx + params.bs_gain_dbi + params.ue_gain_dbi - pl


def received_ris_dbm(xx: np.ndarray, yy: np.ndarray, params: RisParams, mode: str = "optimized") -> np.ndarray:
    ris = ris_position(params)
    d1 = float(np.linalg.norm(BS - ris))
    d2 = np.sqrt((xx - ris[0]) ** 2 + (yy - ris[1]) ** 2)
    bs_to_ris_block = line_blocked_scalar(BS, ris)
    ris_to_ue_block = blocked_mask(ris, xx, yy)
    pl1 = ci_path_loss_db(d1, params, los=not bs_to_ris_block, extra_blockage_db=params.blockage_loss_db if bs_to_ris_block else 0.0)
    pl2 = ci_path_loss_db(d2, params, los=~ris_to_ue_block, extra_blockage_db=np.where(ris_to_ue_block, params.blockage_loss_db, 0.0))

    if mode == "none":
        coherent_gain = -300.0
    elif mode == "random":
        coherent_gain = 10.0 * log10(max(params.ris_elements, 1)) - 7.0
    else:
        coherent_gain = 20.0 * log10(max(params.ris_elements, 1)) + params.ris_phase_efficiency_db

    angle_bias = 3.0 * np.cos(np.arctan2(yy - ris[1], xx - ris[0]) - 0.45)
    return (
        params.tx_power_dbm
        + params.bs_gain_dbi
        + params.ue_gain_dbi
        - pl1
        - pl2
        + coherent_gain
        - params.ris_reflection_loss_db
        + angle_bias
    )


def combine_powers_dbm(*layers: np.ndarray) -> np.ndarray:
    return linear_to_db(sum(db_to_linear(layer) for layer in layers))


def simulation_grid(params: RisParams) -> Dict[str, np.ndarray]:
    axis = np.linspace(-params.grid_span_m, params.grid_span_m, params.grid_points)
    xx, yy = np.meshgrid(axis, axis)
    return {"x": axis, "y": axis, "xx": xx, "yy": yy}


def interference_dbm(xx: np.ndarray, yy: np.ndarray, params: RisParams) -> np.ndarray:
    layers = [received_direct_dbm(bs, xx, yy, params, tx_power_dbm=params.tx_power_dbm - 32.0) for bs in INTERFERERS]
    return combine_powers_dbm(*layers)


def run_scenarios(params: RisParams = RisParams()) -> Dict[str, object]:
    grid = simulation_grid(params)
    xx = grid["xx"]
    yy = grid["yy"]

    direct = received_direct_dbm(BS, xx, yy, params)
    ris_random = received_ris_dbm(xx, yy, params, "random")
    ris_opt = received_ris_dbm(xx, yy, params, "optimized")
    no_ris = direct
    with_random = combine_powers_dbm(direct, ris_random)
    with_opt = combine_powers_dbm(direct, ris_opt)
    inter = interference_dbm(xx, yy, params)
    noise_mw = float(db_to_linear(noise_power_dbm(params)))

    def sinr(signal_dbm: np.ndarray) -> np.ndarray:
        return linear_to_db(db_to_linear(signal_dbm) / (db_to_linear(inter) + noise_mw))

    def throughput(sinr_db: np.ndarray) -> np.ndarray:
        return params.bandwidth_mhz * 0.72 * np.log2(1.0 + db_to_linear(sinr_db))

    sinr_no = sinr(no_ris)
    sinr_random = sinr(with_random)
    sinr_opt = sinr(with_opt)
    blocked = blocked_mask(BS, xx, yy)
    coverage_threshold = -100.0
    covered_no = no_ris > coverage_threshold
    covered_opt = with_opt > coverage_threshold
    blocked_or_weak = blocked | (no_ris <= coverage_threshold)
    covered_rate_no = throughput(sinr_no)[covered_no]
    covered_rate_opt = throughput(sinr_opt)[covered_opt]
    weak_gain = with_opt[blocked_or_weak] - no_ris[blocked_or_weak]

    return {
        **grid,
        "params": params,
        "direct": direct,
        "ris_random": ris_random,
        "ris_opt": ris_opt,
        "no_ris": no_ris,
        "with_random": with_random,
        "with_opt": with_opt,
        "interference": inter,
        "sinr_no": sinr_no,
        "sinr_random": sinr_random,
        "sinr_opt": sinr_opt,
        "throughput_no": throughput(sinr_no),
        "throughput_random": throughput(sinr_random),
        "throughput_opt": throughput(sinr_opt),
        "blocked": blocked,
        "coverage_no": float(np.mean(covered_no)),
        "coverage_opt": float(np.mean(covered_opt)),
        "outage_no": float(np.mean(~covered_no)),
        "outage_opt": float(np.mean(~covered_opt)),
        "median_sinr_no": float(np.median(sinr_no)),
        "median_sinr_opt": float(np.median(sinr_opt)),
        "p5_sinr_no": float(np.percentile(sinr_no, 5)),
        "p5_sinr_opt": float(np.percentile(sinr_opt, 5)),
        "mean_rate_no": float(np.mean(throughput(sinr_no))),
        "mean_rate_opt": float(np.mean(throughput(sinr_opt))),
        "edge_rate_no": float(np.percentile(throughput(sinr_no), 5)),
        "edge_rate_opt": float(np.percentile(throughput(sinr_opt), 5)),
        "covered_edge_rate_no": float(np.percentile(covered_rate_no, 5)) if covered_rate_no.size else 0.0,
        "covered_edge_rate_opt": float(np.percentile(covered_rate_opt, 5)) if covered_rate_opt.size else 0.0,
        "weak_region_mean_gain": float(np.mean(weak_gain)) if weak_gain.size else 0.0,
        "blocked_mean_rsrp_no": float(np.mean(no_ris[blocked])) if np.any(blocked) else float(np.mean(no_ris)),
        "blocked_mean_rsrp_opt": float(np.mean(with_opt[blocked])) if np.any(blocked) else float(np.mean(with_opt)),
        "blocked_mean_rate_no": float(np.mean(throughput(sinr_no)[blocked])) if np.any(blocked) else float(np.mean(throughput(sinr_no))),
        "blocked_mean_rate_opt": float(np.mean(throughput(sinr_opt)[blocked])) if np.any(blocked) else float(np.mean(throughput(sinr_opt))),
    }


def cdf(values: np.ndarray, points: int = 180) -> Tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values.ravel())
    xs = np.linspace(float(sorted_values[0]), float(sorted_values[-1]), points)
    ys = np.searchsorted(sorted_values, xs, side="right") / sorted_values.size
    return xs, ys


def ris_element_sweep(elements: Iterable[int] = (16, 32, 64, 128, 256, 512, 1024)) -> Dict[str, List[float]]:
    result = {
        "elements": [],
        "coverage": [],
        "median_sinr": [],
        "edge_rate_all": [],
        "covered_edge_rate": [],
        "blocked_mean_rate": [],
        "weak_region_gain": [],
    }
    for n in elements:
        params = RisParams(ris_elements=int(n), grid_points=91)
        data = run_scenarios(params)
        result["elements"].append(int(n))
        result["coverage"].append(float(data["coverage_opt"]) * 100.0)
        result["median_sinr"].append(float(data["median_sinr_opt"]))
        result["edge_rate_all"].append(float(data["edge_rate_opt"]))
        result["covered_edge_rate"].append(float(data["covered_edge_rate_opt"]))
        result["blocked_mean_rate"].append(float(data["blocked_mean_rate_opt"]))
        result["weak_region_gain"].append(float(data["weak_region_mean_gain"]))
    return result


def ber_awgn(snr_db: np.ndarray) -> Dict[str, np.ndarray]:
    gamma = db_to_linear(snr_db)
    qpsk = 0.5 * np.vectorize(erfc)(np.sqrt(gamma))
    qam16 = 0.375 * np.vectorize(erfc)(np.sqrt(0.1 * gamma))
    qam64 = (7.0 / 24.0) * np.vectorize(erfc)(np.sqrt(gamma / 42.0))
    return {
        "QPSK": np.maximum(qpsk, 1e-12),
        "16QAM": np.maximum(qam16, 1e-12),
        "64QAM": np.maximum(qam64, 1e-12),
    }


def sample_users(data: Dict[str, object]) -> List[Dict[str, float | str]]:
    xx = data["xx"]
    yy = data["yy"]
    users = [
        ("中心用户", 45.0, 20.0),
        ("遮挡区用户", 133.0, 126.0),
        ("边缘遮挡用户", 189.0, 210.0),
    ]
    rows = []
    for name, ux, uy in users:
        idx = np.unravel_index(np.argmin((xx - ux) ** 2 + (yy - uy) ** 2), xx.shape)
        rows.append(
            {
                "name": name,
                "x": ux,
                "y": uy,
                "rsrp_no": float(data["no_ris"][idx]),
                "rsrp_opt": float(data["with_opt"][idx]),
                "sinr_no": float(data["sinr_no"][idx]),
                "sinr_opt": float(data["sinr_opt"][idx]),
                "rate_no": float(data["throughput_no"][idx]),
                "rate_opt": float(data["throughput_opt"][idx]),
            }
        )
    return rows
