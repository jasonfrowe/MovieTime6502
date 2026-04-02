#!/usr/bin/env python3
"""
generate_showcase_v5.py - Amiga-style bouncing checker ball showcase for MovieTime6502.

Visual plan:
  - Layer 1 (MIDDLE / base): sky + perspective floor with subtle wave motion.
  - Layer 2 (TOP / overlay): bouncing checker sphere with transparency.
  - A soft shadow is baked into the base layer to ground the ball.

Packing plan:
  - Each frame renders full 320x240 indexed images for both layers.
  - Each layer is clustered to a 256-tile dictionary (8x8, 4-bit) using
    MiniBatchKMeans with medoid-like exemplar tiles for stability.
  - Stream packing matches RP6502 layer ordering:
      palette1/layer1 = base (opaque)
      palette2/layer2 = overlay (index 0 transparent)

Usage:
    python tools/generate_showcase_v5.py --output SHOWCASE_V5.BIN
    python tools/generate_showcase_v5.py --output MOVIE.BIN --seconds 60 --fps 24
"""

import argparse
import struct
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


SCREEN_W = 320
SCREEN_H = 240
TILE_W = 8
TILE_H = 8
COLS = SCREEN_W // TILE_W
ROWS = SCREEN_H // TILE_H
NUM_TILES = 256
COLOURS = 16

PALETTE_BYTES = COLOURS * 2
TILES_BYTES = NUM_TILES * TILE_H * (TILE_W // 2)
MAP_BYTES = COLS * ROWS
FRAME_BYTES = (PALETTE_BYTES * 2) + (TILES_BYTES * 2) + (MAP_BYTES * 2)

HEADER_MAGIC = b"MT62"
HEADER_VERSION = 1


def rgb8_to_rgb555(r: int, g: int, b: int, opaque: bool = True) -> int:
    word = ((b >> 3) << 11) | ((g >> 3) << 6) | (r >> 3)
    if opaque:
        word |= 0x0020
    return word


def palette_to_bytes(palette_rgb: np.ndarray, transparent_index0: bool = False) -> bytes:
    out = bytearray()
    for idx, (r, g, b) in enumerate(palette_rgb):
        word = rgb8_to_rgb555(int(r), int(g), int(b), opaque=not (transparent_index0 and idx == 0))
        out += struct.pack("<H", word)
    return bytes(out)


def encode_tile(tile_8x8_idx: np.ndarray) -> bytes:
    out = bytearray(32)
    for row in range(TILE_H):
        for col in range(TILE_W // 2):
            lo = int(tile_8x8_idx[row, col * 2]) & 0xF
            hi = int(tile_8x8_idx[row, col * 2 + 1]) & 0xF
            out[row * 4 + col] = lo | (hi << 4)
    return bytes(out)


def build_tileset(tiles: np.ndarray) -> bytes:
    result = bytearray(TILES_BYTES)
    for tile_id in range(NUM_TILES):
        encoded = encode_tile(tiles[tile_id])
        result[tile_id * 32:(tile_id + 1) * 32] = encoded
    return bytes(result)


def make_base_palette_v5() -> np.ndarray:
    """Dark blue sky to steel/cyan floor palette (opaque)."""
    return np.array([
        (2, 6, 16),
        (6, 12, 26),
        (10, 18, 36),
        (14, 24, 48),
        (20, 34, 62),
        (28, 44, 76),
        (36, 56, 92),
        (46, 70, 108),
        (58, 86, 126),
        (72, 104, 146),
        (88, 124, 166),
        (106, 146, 186),
        (126, 168, 206),
        (148, 190, 224),
        (176, 214, 240),
        (210, 236, 252),
    ], dtype=np.uint8)


def make_overlay_palette_v5() -> np.ndarray:
    """Checker ball palette: dark reds + bright whites. Index 0 is transparent."""
    return np.array([
        (0, 0, 0),      # transparent
        (22, 8, 8),
        (36, 10, 10),
        (52, 14, 14),
        (72, 18, 18),
        (96, 24, 24),
        (122, 30, 30),
        (150, 40, 40),
        (182, 52, 52),
        (218, 70, 70),
        (84, 84, 84),
        (112, 112, 112),
        (146, 146, 146),
        (182, 182, 182),
        (220, 220, 220),
        (252, 252, 252),
    ], dtype=np.uint8)


def find_tiles(indexed_img: np.ndarray, n_tiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster 1200 8x8 patches to n_tiles dictionary and tile map."""
    tiles = []
    for r in range(ROWS):
        for c in range(COLS):
            patch = indexed_img[r*TILE_H:(r+1)*TILE_H, c*TILE_W:(c+1)*TILE_W].astype(np.float32)
            tiles.append(patch.flatten())
    tiles_arr = np.array(tiles)

    unique_tiles, inverse_indices = np.unique(tiles_arr, axis=0, return_inverse=True)
    if len(unique_tiles) <= n_tiles:
        tile_dict = np.zeros((n_tiles, TILE_H, TILE_W), dtype=np.uint8)
        tile_dict[:len(unique_tiles)] = unique_tiles.reshape(-1, TILE_H, TILE_W).astype(np.uint8)
        return tile_dict, inverse_indices.reshape(ROWS, COLS).astype(np.uint8)

    km = MiniBatchKMeans(n_clusters=n_tiles, n_init=3, max_iter=60, random_state=42, batch_size=512)
    km.fit(tiles_arr)
    tile_ids = km.predict(tiles_arr)

    centres = km.cluster_centers_.astype(np.float32)
    tile_dict = np.zeros((n_tiles, TILE_H, TILE_W), dtype=np.uint8)
    for k in range(n_tiles):
        members = np.where(tile_ids == k)[0]
        if members.size == 0:
            tile_dict[k] = np.clip(np.round(centres[k].reshape(TILE_H, TILE_W)), 0, 15).astype(np.uint8)
            continue
        member_tiles = tiles_arr[members]
        d2 = np.sum((member_tiles - centres[k]) ** 2, axis=1)
        best_idx = members[int(np.argmin(d2))]
        tile_dict[k] = tiles_arr[best_idx].reshape(TILE_H, TILE_W).astype(np.uint8)

    tile_map = tile_ids.reshape(ROWS, COLS).astype(np.uint8)
    return tile_dict, tile_map


def generate_base_highres_v5(frame_idx: int, ball_x: float, ball_y: float, ball_r: float) -> np.ndarray:
    """Render sky, perspective floor, and shadow into a 320x240 index image (0-15)."""
    y = np.arange(SCREEN_H, dtype=np.float32)[:, None]
    x = np.arange(SCREEN_W, dtype=np.float32)[None, :]

    horizon = 94.0 + 3.0 * np.sin(frame_idx * 0.012)

    # Sky gradient with subtle horizontal shimmer.
    sky_t = np.clip((horizon - y) / max(horizon, 1.0), 0.0, 1.0)
    sky_wave = 0.5 + 0.5 * np.sin(x * 0.03 + frame_idx * 0.05)
    sky_idx = 2.0 + 4.0 * (1.0 - sky_t) + 1.1 * sky_wave

    # Perspective floor checker with slight wave drift.
    dy = np.maximum(y - horizon, 1.0)
    depth = 120.0 / dy
    wx = (x - SCREEN_W * 0.5) * depth
    wz = (180.0 * depth) + frame_idx * 0.9
    floor_checker = ((np.floor(wx * 0.32) + np.floor(wz * 0.42)) % 2.0)
    floor_grad = np.clip((y - horizon) / (SCREEN_H - horizon + 1.0), 0.0, 1.0)
    floor_idx = 6.0 + 6.0 * floor_grad + floor_checker * 2.2

    base = np.where(y < horizon, sky_idx, floor_idx)

    # Contact shadow ellipse projected on the floor.
    shadow_cy = min(SCREEN_H - 8.0, ball_y + ball_r * 1.08)
    shadow_rx = ball_r * (1.2 + 0.2 * np.cos(frame_idx * 0.04))
    shadow_ry = max(5.0, ball_r * 0.24)
    dist = ((x - ball_x) / max(shadow_rx, 1.0))**2 + ((y - shadow_cy) / max(shadow_ry, 1.0))**2
    shadow = np.clip(1.0 - dist, 0.0, 1.0)

    # Shadow strength increases as the ball approaches the floor.
    floor_y = SCREEN_H - 44.0
    near_floor = 1.0 - np.clip((floor_y - ball_y) / 90.0, 0.0, 1.0)
    base -= shadow * (1.8 + 2.8 * near_floor)

    return np.clip(np.rint(base), 0, 15).astype(np.uint8)


def generate_overlay_highres_v5(frame_idx: int) -> tuple[np.ndarray, float, float, float]:
    """Render bouncing checker sphere into overlay index image (0 transparent, 1-15 visible)."""
    img = np.zeros((SCREEN_H, SCREEN_W), dtype=np.uint8)

    t = frame_idx / 24.0
    # Horizontal motion + slight drift.
    ball_x = 160.0 + 110.0 * np.sin(t * 1.25)

    floor_y = SCREEN_H - 44.0
    # Bounce curve: abs(sin) gives repeated parabolic-like floor contacts.
    bounce = np.abs(np.sin(t * 1.95))
    ball_y = floor_y - (20.0 + 82.0 * bounce)

    # Squash at contact and stretch mid-air.
    contact = np.exp(-((bounce - 0.0) / 0.17)**2)
    stretch = np.exp(-((bounce - 1.0) / 0.25)**2)
    ball_r = 28.0 * (1.0 - 0.10 * contact + 0.06 * stretch)

    x = np.arange(SCREEN_W, dtype=np.float32)
    y = np.arange(SCREEN_H, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    dx = X - ball_x
    dy = Y - ball_y
    r2 = (dx * dx + dy * dy) / max(ball_r * ball_r, 1.0)
    inside = r2 <= 1.0
    if not np.any(inside):
        return img, ball_x, ball_y, ball_r

    nx = dx / max(ball_r, 1.0)
    ny = dy / max(ball_r, 1.0)
    nz = np.zeros_like(nx)
    nz[inside] = np.sqrt(np.maximum(0.0, 1.0 - r2[inside]))

    # Rotate checker texture around Y and tilt slightly with bounce.
    ang = t * 3.8
    ca, sa = np.cos(ang), np.sin(ang)
    xr = ca * nx + sa * nz
    zr = -sa * nx + ca * nz
    yr = 0.93 * ny + 0.07 * np.sin(t * 2.4)

    u = np.arctan2(zr, xr) / (2.0 * np.pi) + 0.5
    v = np.arcsin(np.clip(yr, -1.0, 1.0)) / np.pi + 0.5
    checks = ((np.floor(u * 8.0) + np.floor(v * 8.0)).astype(np.int32) & 1) == 0

    # Lighting model with strong highlight and rim darkening.
    lx, ly, lz = -0.35, -0.55, 0.76
    lnorm = (lx * lx + ly * ly + lz * lz) ** 0.5
    lx /= lnorm; ly /= lnorm; lz /= lnorm
    diff = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    rim = np.clip(nz, 0.0, 1.0)
    light = np.clip(0.18 + 0.92 * diff * (0.65 + 0.35 * rim), 0.0, 1.0)

    # Checker colors use dedicated red and white ramps in indices 1..15.
    red_ramp = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
    wht_ramp = np.array([10, 11, 12, 13, 14, 15], dtype=np.uint8)

    rlev = np.clip((light * (len(red_ramp) - 1)).astype(np.int32), 0, len(red_ramp) - 1)
    wlev = np.clip((light * (len(wht_ramp) - 1)).astype(np.int32), 0, len(wht_ramp) - 1)

    idx = np.where(checks, wht_ramp[wlev], red_ramp[rlev]).astype(np.uint8)

    # Specular flash near top-left of sphere.
    hx, hy, hz = -0.24, -0.30, 0.92
    hnorm = (hx * hx + hy * hy + hz * hz) ** 0.5
    hx /= hnorm; hy /= hnorm; hz /= hnorm
    spec = np.power(np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0), 36.0)
    idx[spec > 0.35] = 15

    img[inside] = idx[inside]
    return img, ball_x, ball_y, ball_r


def build_frame(frame_idx: int, fps: int) -> bytes:
    _ = fps  # animation speed is tuned for 24fps; keep arg for CLI compatibility
    palette_overlay = make_overlay_palette_v5()
    overlay_img, bx, by, br = generate_overlay_highres_v5(frame_idx)
    base_img = generate_base_highres_v5(frame_idx, bx, by, br)

    base_tiles, base_map = find_tiles(base_img, NUM_TILES)
    over_tiles, over_map = find_tiles(overlay_img, NUM_TILES)

    palette_base = make_base_palette_v5()

    return (
        palette_to_bytes(palette_base)                             # base palette    → Layer 1 slot
        + palette_to_bytes(palette_overlay, transparent_index0=True)  # overlay palette → Layer 2 slot
        + build_tileset(over_tiles)                               # overlay tiles   → Layer 2 slot
        + build_tileset(base_tiles)                               # base tiles      → Layer 1 slot
        + bytes(over_map.flatten())                               # overlay map     → Layer 2 slot
        + bytes(base_map.flatten())                               # base map        → Layer 1 slot
    )


def generate_stream(output_path: Path, seconds: float, fps: int) -> None:
    frame_count = int(round(seconds * fps))
    if frame_count <= 0:
        raise ValueError("seconds * fps must produce at least one frame")

    print("Generating V5 Amiga-style bouncing ball showcase")
    print(f"Frames: {frame_count}")
    print(f"FPS: {fps}")
    print(f"Duration: {seconds:.2f}s")
    print(f"Frame payload: {FRAME_BYTES:,} bytes")

    header = struct.pack(
        "<4sBBHHHHI",
        HEADER_MAGIC,
        HEADER_VERSION,
        fps,
        SCREEN_W,
        SCREEN_H,
        TILE_W,
        TILE_H,
        frame_count,
    )

    t0 = time.time()
    with output_path.open("wb") as fout:
        fout.write(header)
        for frame_idx in range(frame_count):
            fout.write(build_frame(frame_idx, fps))
            if frame_idx % fps == 0 or frame_idx == frame_count - 1:
                elapsed = time.time() - t0
                rate = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
                pct = (frame_idx + 1) * 100.0 / frame_count
                print(f"  [{pct:5.1f}%] frame {frame_idx + 1}/{frame_count}  gen {rate:.1f} fr/s", end="\r")

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"Output: {output_path} ({output_path.stat().st_size / 1024.0 / 1024.0:.1f} MiB)")
    print("Copy or rename this file to MOVIE.BIN to play it on the RP6502 demo ROM.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V5 Amiga-style bouncing ball MT62 stream")
    parser.add_argument("--output", default="SHOWCASE_V5.BIN", help="Output stream filename")
    parser.add_argument("--seconds", type=float, default=60.0, help="Length of the showcase in seconds")
    parser.add_argument("--fps", type=int, default=24, help="Playback FPS (default: 24)")
    args = parser.parse_args()

    generate_stream(Path(args.output), args.seconds, args.fps)


if __name__ == "__main__":
    main()
