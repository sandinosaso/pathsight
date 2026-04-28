"""Heatmap coloring and PNG/base64 encoding."""

from __future__ import annotations

import base64
from typing import Tuple

import cv2
import numpy as np


def bytes_to_rgb_u8(raw: bytes) -> np.ndarray:
    """Decode upload bytes to an RGB uint8 ndarray (H, W, 3)."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to decode uploaded image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def array_to_png_base64(rgb_u8: np.ndarray) -> str:
    """Encode RGB image as PNG base64 string (no data URL prefix)."""
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def bytes_to_png_base64(raw: bytes) -> str:
    """Decode image bytes and re-encode as PNG base64 string."""
    return array_to_png_base64(bytes_to_rgb_u8(raw))


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1].  Returns zeros when the input is flat."""
    h = heatmap.astype(np.float32)
    lo, hi = float(h.min()), float(h.max())
    if hi <= lo:
        return np.zeros_like(h)
    return (h - lo) / (hi - lo)


def colorize_heatmap(heatmap_01: np.ndarray) -> np.ndarray:
    """Float [0, 1] heatmap -> BGR uint8 jet image."""
    u8 = (np.clip(heatmap_01, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def heatmap_to_rgb_u8(heatmap: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Resize the (H', W') heatmap to (H, W) and return as RGB uint8."""
    h01 = normalize_heatmap(heatmap)
    bgr = colorize_heatmap(h01)
    h, w = out_hw
    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def blend_overlay(
    original_rgb_u8: np.ndarray,
    heatmap_rgb_u8: np.ndarray,
    *,
    alpha: float = 0.4,
) -> np.ndarray:
    """Alpha-blend a heatmap (RGB) over an original image (RGB).

    Both inputs must be RGB uint8.  The blend is performed in BGR
    (OpenCV's native space) for consistency, then converted back to RGB.
    ``alpha`` is the heatmap's contribution.
    """
    orig_bgr = cv2.cvtColor(original_rgb_u8, cv2.COLOR_RGB2BGR)
    heat_bgr = cv2.cvtColor(heatmap_rgb_u8, cv2.COLOR_RGB2BGR)
    h, w = orig_bgr.shape[:2]
    if heat_bgr.shape[:2] != (h, w):
        heat_bgr = cv2.resize(heat_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    blended = cv2.addWeighted(orig_bgr, 1 - alpha, heat_bgr, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
