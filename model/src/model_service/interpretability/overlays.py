"""Heatmap coloring and PNG/base64 encoding."""

from __future__ import annotations

import base64

import cv2
import numpy as np

def array_to_png_base64(rgb_u8: np.ndarray) -> str:
    """Encode RGB image as PNG base64 string (no data URL prefix)."""
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")

def bytes_to_png_base64(raw: bytes) -> str:
    """Decode image bytes and re-encode as PNG base64 string."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to decode uploaded image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return array_to_png_base64(rgb)
