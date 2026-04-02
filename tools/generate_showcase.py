#!/usr/bin/env python3
"""
generate_showcase.py - Procedural MT62 showcase generator for MovieTime6502.

This generates a synthetic demo stream that is designed to show off the RP6502
movie player format rather than encode source video. Both tile layers use the
available tile slots and both palettes rotate over time.

Visual design:
  - Layer 2: faux 3D tunnel walls with depth shading and angular striping.
  - Layer 1: transparent HUD rings and sweep lines floating over the tunnel.
  - Palettes rotate every frame so motion comes from both tile updates and
    per-frame palette changes.

Usage:
    python tools/generate_showcase.py --output SHOWCASE.BIN
    python tools/generate_showcase.py --output MOVIE.BIN --seconds 20 --fps 24
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


def make_base_palette(frame_idx: int, fps: int) -> np.ndarray:
    phase = frame_idx / max(fps, 1)
    palette = np.zeros((COLOURS, 3), dtype=np.uint8)
    for idx in range(COLOURS):
        t = idx / (COLOURS - 1)
        hue = 0.60 + phase * 0.08 + t * 0.18
        sat = 0.90 - t * 0.20
        val = 0.05 + (t ** 1.3) * 0.95
        palette[idx] = hsv_rgb_bytes(hue, sat, val)
    return palette


def make_overlay_palette(frame_idx: int, fps: int) -> np.ndarray:
    phase = frame_idx / max(fps, 1)
    palette = np.zeros((COLOURS, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    for idx in range(1, COLOURS):
        t = idx / (COLOURS - 1)
        hue = 0.02 + phase * 0.14 + t * 0.05
        sat = 0.15 + t * 0.50
        val = 0.30 + t * 0.70
        if idx >= 12:
            sat *= 0.30
            val = min(1.0, val + 0.14)
        palette[idx] = hsv_rgb_bytes(hue, sat, val)
    return palette


def tunnel_bins(px: np.ndarray, py: np.ndarray, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = frame_idx * 0.06
    cx = SCREEN_W * 0.50 + np.sin(phase * 0.80) * 18.0
    cy = SCREEN_H * 0.50 + np.cos(phase * 0.65) * 12.0

    dx = px - cx
    dy = py - cy
    radius = np.hypot(dx * 1.08, dy * 0.92) + 1.0
    angle = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)

    depth = (350.0 / radius) + phase * 5.0
    angle_phase = angle + phase * 0.18

    angle_bin = np.floor((angle_phase % 1.0) * 16.0).astype(np.uint8)
    depth_bin = np.floor(np.mod(depth, 16.0)).astype(np.uint8)
    brightness = np.clip(15.0 - radius / 13.0 + 3.0 * np.sin(depth * 0.70), 0.0, 15.0)

    return angle_bin, depth_bin, np.rint(brightness).astype(np.uint8)


def generate_base_tiles(frame_idx: int) -> np.ndarray:
    x = np.arange(TILE_W, dtype=np.float32)[None, :]
    y = np.arange(TILE_H, dtype=np.float32)[:, None]
    tiles = np.empty((NUM_TILES, TILE_H, TILE_W), dtype=np.uint8)

    for depth_bin in range(16):
        for angle_bin in range(16):
            tile_id = (depth_bin << 4) | angle_bin
            local_phase = frame_idx * 0.10 + angle_bin * 0.35 + depth_bin * 0.22
            stripe = np.sin((x + angle_bin * 0.7) * 0.95 + local_phase)
            mortar = np.cos((y + depth_bin * 0.8) * 1.05 - local_phase * 0.8)
            diagonal = np.sin((x + y) * 0.45 + local_phase * 1.4)
            field = 6.0 + depth_bin * 0.52 + stripe * 3.8 + mortar * 2.2 + diagonal * 1.4
            tiles[tile_id] = np.clip(np.rint(field), 0.0, 15.0).astype(np.uint8)

    return tiles


def generate_overlay_tiles(frame_idx: int) -> np.ndarray:
    x = np.arange(TILE_W, dtype=np.float32)[None, :]
    y = np.arange(TILE_H, dtype=np.float32)[:, None]
    tiles = np.zeros((NUM_TILES, TILE_H, TILE_W), dtype=np.uint8)

    for ring_bin in range(16):
        for spoke_bin in range(16):
            tile_id = (ring_bin << 4) | spoke_bin
            local_phase = frame_idx * 0.12 + ring_bin * 0.20 - spoke_bin * 0.18
            arc = np.sin((x - 3.5) * 0.85 + spoke_bin * 0.55 + local_phase)
            scan = np.cos((y - 3.5) * 0.80 + ring_bin * 0.60 - local_phase)
            glow = np.sin((x + y) * 0.35 + local_phase * 1.8)
            field = 0.55 * arc + 0.55 * scan + 0.35 * glow
            active = field > 0.55
            strength = np.clip(np.rint((field - 0.55) * 20.0) + 4.0 + ring_bin * 0.25, 1.0, 15.0)
            tiles[tile_id] = np.where(active, strength, 0).astype(np.uint8)

    return tiles


def generate_maps(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    px = (np.arange(COLS, dtype=np.float32)[None, :] + 0.5) * TILE_W
    py = (np.arange(ROWS, dtype=np.float32)[:, None] + 0.5) * TILE_H

    angle_bin, depth_bin, brightness = tunnel_bins(px, py, frame_idx)
    base_map = ((depth_bin << 4) | angle_bin).astype(np.uint8)

    phase = frame_idx * 0.08
    cx = SCREEN_W * 0.50
    cy = SCREEN_H * 0.50
    dx = px - cx
    dy = py - cy
    radius = np.hypot(dx, dy)
    angle = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)

    ring_bin = np.floor(np.mod(radius * 0.12 - phase * 5.0, 16.0)).astype(np.uint8)
    spoke_bin = np.floor(np.mod(angle * 16.0 + phase * 3.0, 16.0)).astype(np.uint8)

    hud_mask = (brightness > 2) & (((ring_bin + spoke_bin + (frame_idx & 0xFF)) & 3) != 0)
    overlay_map = np.where(hud_mask, (ring_bin << 4) | spoke_bin, 0).astype(np.uint8)

    return base_map, overlay_map


def build_frame(frame_idx: int, fps: int) -> bytes:
    palette1 = make_overlay_palette(frame_idx, fps)
    palette2 = make_base_palette(frame_idx, fps)
    tiles1 = generate_overlay_tiles(frame_idx)
    tiles2 = generate_base_tiles(frame_idx)
    map2, map1 = generate_maps(frame_idx)

    return (
        palette_to_bytes(palette1, transparent_index0=True)
        + palette_to_bytes(palette2)
        + build_tileset(tiles2)
        + build_tileset(tiles1)
        + bytes(map2.flatten())
        + bytes(map1.flatten())
    )


def generate_stream(output_path: Path, seconds: float, fps: int) -> None:
    frame_count = int(round(seconds * fps))
    if frame_count <= 0:
        raise ValueError("seconds * fps must produce at least one frame")

    print("Generating procedural MT62 tunnel showcase")
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
    parser = argparse.ArgumentParser(description="Generate a procedural MT62 showcase stream")
    parser.add_argument("--output", default="SHOWCASE.BIN", help="Output stream filename")
    parser.add_argument("--seconds", type=float, default=20.0, help="Length of the showcase in seconds")
    parser.add_argument("--fps", type=int, default=24, help="Playback FPS (default: 24)")
    args = parser.parse_args()

    generate_stream(Path(args.output), args.seconds, args.fps)


if __name__ == "__main__":
    main()