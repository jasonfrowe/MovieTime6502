#!/usr/bin/env python3
"""
generate_showcase_v2.1.py - Procedural MT62 fractal showcase for MovieTime6502.

This generates a synthetic demo stream highlighting the 400KB/s bandwidth capabilities
of the RP6502 MT62 format.

Visual design:
  - Layer 2 (Base): A morphing, zooming Julia Set evaluated at full 320x240
    resolution. K-Means clustering is used to pack the resulting detail into
    the 256 tile limitation dynamically per frame.
  - Layer 1 (Overlay): A zooming bitwise XOR fractal evaluated at full 320x240
    resolution, clustered into 256 tiles to float cyber-grid structures over
    the base layer.
  - Palettes: Both layers undergo furious saturated color cycling.

Usage:
    python tools/generate_showcase_v2.1.py --output SHOWCASE_V2.1.BIN
    python tools/generate_showcase_v2.1.py --output MOVIE.BIN --seconds 60 --fps 24
"""

import argparse
import colorsys
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
        start = tile_id * 32
        result[start:start + 32] = encoded
    return bytes(result)


def hsv_rgb_bytes(hue: float, sat: float, val: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, max(0.0, min(1.0, sat)), max(0.0, min(1.0, val)))
    return int(r * 255.0), int(g * 255.0), int(b * 255.0)


def make_base_palette_v2(frame_idx: int, fps: int) -> np.ndarray:
    phase = frame_idx / max(fps, 1)
    palette = np.zeros((COLOURS, 3), dtype=np.uint8)
    for idx in range(COLOURS):
        t = idx / (COLOURS - 1)
        hue = (t * 1.5 + phase * 0.8) % 1.0
        sat = 0.85 + 0.15 * np.sin(t * np.pi)
        val = 0.4 + 0.6 * np.cos(t * 2 * np.pi)
        palette[idx] = hsv_rgb_bytes(hue, sat, val)
    return palette


def make_overlay_palette_v2(frame_idx: int, fps: int) -> np.ndarray:
    phase = frame_idx / max(fps, 1)
    palette = np.zeros((COLOURS, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    for idx in range(1, COLOURS):
        t = idx / (COLOURS - 1)
        hue = (t * 0.8 + phase * 1.2) % 1.0
        sat = 0.9 - 0.2 * t
        val = 0.8 + 0.2 * np.sin(phase * 5.0 + t * np.pi)
        palette[idx] = hsv_rgb_bytes(hue, sat, val)
    return palette


def find_tiles(indexed_img: np.ndarray, n_tiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the 1200 8x8 blocks of an image down to an optimal n_tiles dictionary."""
    tiles = []
    for r in range(ROWS):
        for c in range(COLS):
            patch = indexed_img[r*TILE_H:(r+1)*TILE_H, c*TILE_W:(c+1)*TILE_W].astype(np.float32)
            tiles.append(patch.flatten())
    tiles_arr = np.array(tiles)

    # Fast path if the image naturally has <= n_tiles unique blocks
    unique_tiles, inverse_indices = np.unique(tiles_arr, axis=0, return_inverse=True)
    if len(unique_tiles) <= n_tiles:
        tile_dict = np.zeros((n_tiles, TILE_H, TILE_W), dtype=np.uint8)
        tile_dict[:len(unique_tiles)] = unique_tiles.reshape(-1, TILE_H, TILE_W).astype(np.uint8)
        return tile_dict, inverse_indices.reshape(ROWS, COLS).astype(np.uint8)

    km = MiniBatchKMeans(n_clusters=n_tiles, n_init=3, max_iter=50, random_state=42, batch_size=512)
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


def generate_base_highres(frame_idx: int) -> np.ndarray:
    # --- BASE LAYER: Full Resolution Morphing Julia Set ---
    j_zoom = 1.2 + np.sin(frame_idx * 0.015) * 0.6
    j_ox = np.cos(frame_idx * 0.02) * 0.3
    j_oy = np.sin(frame_idx * 0.025) * 0.3

    cx = np.linspace(-1.8 * j_zoom, 1.8 * j_zoom, SCREEN_W) + j_ox
    cy = np.linspace(-1.35 * j_zoom, 1.35 * j_zoom, SCREEN_H) + j_oy
    X, Y = np.meshgrid(cx, cy)
    Z = X + 1j * Y

    # Classic morphing parameter C
    phase = frame_idx * 0.04
    C = 0.7885 * np.exp(1j * phase)

    iters = np.zeros_like(Z, dtype=np.float32)
    for _ in range(24):
        mask = np.abs(Z) < 10.0
        Z[mask] = Z[mask]**2 + C
        iters[mask] += 1

    # Smooth iteration count
    abs_Z = np.abs(Z)
    mask_escaped = abs_Z > 1.0
    nu = np.copy(iters)
    nu[mask_escaped] = iters[mask_escaped] - np.log2(np.log2(abs_Z[mask_escaped] + 1e-10))

    return np.floor(np.mod(nu * 3.0 - frame_idx * 0.5, 16)).astype(np.uint8)


def generate_overlay_highres(frame_idx: int) -> np.ndarray:
    # --- OVERLAY LAYER: Full Resolution Zooming Bitwise XOR Fractal ---
    over_zoom = 0.5 + np.abs(np.sin(frame_idx * 0.01)) * 4.0
    over_ox = frame_idx * 2.0
    over_oy = frame_idx * 1.5

    X_over = (np.arange(SCREEN_W) - SCREEN_W/2) * over_zoom + over_ox
    Y_over = (np.arange(SCREEN_H) - SCREEN_H/2) * over_zoom + over_oy
    Xg, Yg = np.meshgrid(X_over, Y_over)

    X_int = Xg.astype(np.int32)
    Y_int = Yg.astype(np.int32)

    p1 = np.mod(X_int ^ Y_int, 16)
    p2 = np.mod((X_int & Y_int) + int(frame_idx * 0.5), 16)
    val = p1 ^ p2
    
    mask = (val % 7) < 2
    return np.where(mask, val % 15 + 1, 0).astype(np.uint8)


def build_frame(frame_idx: int, fps: int) -> bytes:
    palette1 = make_overlay_palette_v2(frame_idx, fps)
    palette2 = make_base_palette_v2(frame_idx, fps)
    
    base_img = generate_base_highres(frame_idx)
    tiles2, map2 = find_tiles(base_img, NUM_TILES)
    
    over_img = generate_overlay_highres(frame_idx)
    tiles1, map1 = find_tiles(over_img, NUM_TILES)

    return (
        palette_to_bytes(palette2)                              # base palette    → Layer 1 slot (MIDDLE)
        + palette_to_bytes(palette1, transparent_index0=True)  # overlay palette → Layer 2 slot (TOP)
        + build_tileset(tiles1)                                # overlay tiles   → Layer 2 slot (TOP)
        + build_tileset(tiles2)                                # base tiles      → Layer 1 slot (MIDDLE)
        + bytes(map1.flatten())                                # overlay map     → Layer 2 slot (TOP)
        + bytes(map2.flatten())                                # base map        → Layer 1 slot (MIDDLE)
    )


def generate_stream(output_path: Path, seconds: float, fps: int) -> None:
    frame_count = int(round(seconds * fps))
    if frame_count <= 0:
        raise ValueError("seconds * fps must produce at least one frame")

    print("Generating V2.1 Procedural MT62 Fractal Showcase")
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
    parser = argparse.ArgumentParser(description="Generate V2.1 procedural MT62 fractal stream")
    parser.add_argument("--output", default="SHOWCASE_V2.1.BIN", help="Output stream filename")
    parser.add_argument("--seconds", type=float, default=60.0, help="Length of the showcase in seconds")
    parser.add_argument("--fps", type=int, default=24, help="Playback FPS (default: 24)")
    args = parser.parse_args()

    generate_stream(Path(args.output), args.seconds, args.fps)


if __name__ == "__main__":
    main()