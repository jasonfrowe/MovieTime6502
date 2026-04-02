#!/usr/bin/env python3
"""
generate_showcase_v2.py - Procedural MT62 fractal showcase for MovieTime6502.

This generates a synthetic demo stream highlighting the 400KB/s bandwidth capabilities
of the RP6502 MT62 format.

Visual design:
  - Layer 2 (Base): A morphing, zooming Julia Set. Evaluated at 40x30 to stay within
    the 256 tile limit, but reconstructed at 320x240 using the tiles as a directional
    contour/gradient dictionary.
  - Layer 1 (Overlay): A zooming bitwise XOR fractal (Sierpinski/Munching Squares).
    Uses the 256 tiles to render geometric "cyber/circuit" symbols that rotate and
    pulse with color.
  - Palettes: Both layers undergo furious saturated color cycling.

Usage:
    python tools/generate_showcase_v2.py --output SHOWCASE_V2.BIN
    python tools/generate_showcase_v2.py --output MOVIE.BIN --seconds 20 --fps 24
"""

import argparse
import colorsys
import struct
import time
from pathlib import Path

import numpy as np

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


def generate_maps_v2(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    # --- BASE MAP: Morphing Julia Set Distance Field ---
    # Camera pan and zoom
    j_zoom = 1.2 + np.sin(frame_idx * 0.015) * 0.4
    j_ox = np.cos(frame_idx * 0.02) * 0.3
    j_oy = np.sin(frame_idx * 0.025) * 0.3

    cx = np.linspace(-1.8 * j_zoom, 1.8 * j_zoom, COLS) + j_ox
    cy = np.linspace(-1.35 * j_zoom, 1.35 * j_zoom, ROWS) + j_oy
    X, Y = np.meshgrid(cx, cy)
    Z = X + 1j * Y

    # Classic morphing parameter C
    phase = frame_idx * 0.04
    C = 0.7885 * np.exp(1j * phase)

    iters = np.zeros_like(Z, dtype=np.float32)
    for _ in range(16):
        mask = np.abs(Z) < 10.0
        Z[mask] = Z[mask]**2 + C
        iters[mask] += 1

    # Smooth iteration count
    abs_Z = np.abs(Z)
    mask_escaped = abs_Z > 1.0
    nu = np.copy(iters)
    nu[mask_escaped] = iters[mask_escaped] - np.log2(np.log2(abs_Z[mask_escaped] + 1e-10))

    p1_base = np.floor(np.mod(nu * 3.0 - frame_idx * 0.5, 16)).astype(np.uint8)
    p2_base = np.floor(np.mod(np.angle(Z) * 8 / np.pi + frame_idx * 0.3, 16)).astype(np.uint8)
    base_map = (p1_base << 4) | p2_base

    # --- OVERLAY MAP: Zooming Bitwise XOR Fractal ---
    over_zoom = 0.5 + np.abs(np.sin(frame_idx * 0.01)) * 4.0
    over_ox = frame_idx * 0.5
    over_oy = frame_idx * 0.3

    X_over = (np.arange(COLS) - COLS/2) * over_zoom + over_ox
    Y_over = (np.arange(ROWS) - ROWS/2) * over_zoom + over_oy
    Xg, Yg = np.meshgrid(X_over, Y_over)

    X_int = Xg.astype(np.int32)
    Y_int = Yg.astype(np.int32)

    p1_over = np.mod(X_int ^ Y_int, 16).astype(np.uint8)
    p2_over = np.mod((X_int & Y_int) + frame_idx * 0.5, 16).astype(np.uint8)
    overlay_map = (p1_over << 4) | p2_over

    return base_map, overlay_map


def generate_base_tiles_v2(frame_idx: int) -> np.ndarray:
    """High frequency contour generator for the Julia set macro blocks."""
    tiles = np.empty((NUM_TILES, TILE_H, TILE_W), dtype=np.uint8)
    x = np.arange(TILE_W, dtype=np.float32)[None, :]
    y = np.arange(TILE_H, dtype=np.float32)[:, None]

    for p1 in range(16):
        for p2 in range(16):
            tile_id = (p1 << 4) | p2
            theta = p2 * 2.0 * np.pi / 16.0
            # Draw topographical contour lines oriented by the Julia set gradient
            wave = np.sin((x - 3.5) * np.cos(theta) + (y - 3.5) * np.sin(theta) + p1 * 0.4)
            color = p1 + wave * 2.5
            tiles[tile_id] = np.clip(np.mod(np.rint(color), 16), 0, 15).astype(np.uint8)

    return tiles


def generate_overlay_tiles_v2(frame_idx: int) -> np.ndarray:
    """Geometric cyber grid symbols to populate the XOR fractal blocks."""
    tiles = np.zeros((NUM_TILES, TILE_H, TILE_W), dtype=np.uint8)
    x = np.arange(TILE_W)
    y = np.arange(TILE_H)
    X_tile, Y_tile = np.meshgrid(x, y)

    for p1 in range(16):
        for p2 in range(16):
            tile_id = (p1 << 4) | p2
            shape = p1 % 8
            color = max(1, p2) # 0 is transparent

            val = np.zeros((8, 8), dtype=np.uint8)
            if shape == 1:   val[3:5, 3:5] = color                        # Dot
            elif shape == 2: val[3:5, :] = color                          # Horiz line
            elif shape == 3: val[:, 3:5] = color                          # Vert line
            elif shape == 4: val[3:5, :] = color; val[:, 3:5] = color     # Cross
            elif shape == 5: val[X_tile == Y_tile] = color; val[X_tile == (7 - Y_tile)] = color # X
            elif shape == 6: val[1:7, 1:7] = color; val[2:6, 2:6] = 0     # Box
            elif shape == 7: val[(X_tile + Y_tile) % 2 == 0] = color      # Checkerboard

            # Sparsify the cyber grid to let the Julia set bleed through
            if (p1 ^ p2) % 3 == 0:
                tiles[tile_id] = val

    return tiles


def build_frame(frame_idx: int, fps: int) -> bytes:
    palette1 = make_overlay_palette_v2(frame_idx, fps)
    palette2 = make_base_palette_v2(frame_idx, fps)
    tiles1 = generate_overlay_tiles_v2(frame_idx)
    tiles2 = generate_base_tiles_v2(frame_idx)
    map2, map1 = generate_maps_v2(frame_idx)

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

    print("Generating V2 Procedural MT62 Fractal Showcase")
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
    parser = argparse.ArgumentParser(description="Generate V2 procedural MT62 fractal stream")
    parser.add_argument("--output", default="SHOWCASE_V2.BIN", help="Output stream filename")
    parser.add_argument("--seconds", type=float, default=60.0, help="Length of the showcase in seconds")
    parser.add_argument("--fps", type=int, default=24, help="Playback FPS (default: 24)")
    args = parser.parse_args()

    generate_stream(Path(args.output), args.seconds, args.fps)


if __name__ == "__main__":
    main()