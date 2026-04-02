#!/usr/bin/env python3
"""
generate_showcase_v4.py - Procedural MT62 fractal showcase for MovieTime6502.

Visual design:
  - Layer 1 (Base): A 3D perspective floor grid and soft dynamic shadow.
  - Layer 2 (Overlay): A fully 3D rendered, rotating, bouncing "Amiga" style
    checkerboard sphere (Boing Ball) with Lambertian lighting and specular highlights.

Because the ball's footprint naturally consumes fewer than 256 8x8 blocks, the
clustering logic will automatically trigger a lossless fast-path and pack the
overlay instantly!

Usage:
    python tools/generate_showcase_v4.py --output SHOWCASE_V4.BIN
    python tools/generate_showcase_v4.py --output MOVIE.BIN --seconds 60 --fps 24
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
        start = tile_id * 32
        result[start:start + 32] = encoded
    return bytes(result)


def make_palettes(frame_idx: int, fps: int) -> tuple[np.ndarray, np.ndarray]:
    # --- BASE PALETTE (Layer 1) ---
    pal_base = np.zeros((16, 3), dtype=np.uint8)
    # 0: Deep synthwave background
    pal_base[0] = (15, 0, 30)
    # 1-7: Floor grid lines (magenta fading to dark background)
    for i in range(1, 8):
        t = i / 7.0
        pal_base[i] = (int(15 + 200 * t), int(0 + 40 * t), int(30 + 180 * t))
    # 8-15: Floor shadow (fading from dark purple to deep black)
    for i in range(8, 16):
        t = (i - 8) / 7.0  # 0.0 to 1.0
        pal_base[i] = (int(15 * (1 - t)), 0, int(30 * (1 - t)))

    # --- OVERLAY PALETTE (Layer 2) ---
    pal_over = np.zeros((16, 3), dtype=np.uint8)
    pal_over[0] = (0, 0, 0)  # Index 0 is transparent!
    # 1-7: Shaded Red for Boing Ball
    for i in range(1, 8):
        t = i / 7.0
        pal_over[i] = (int(255 * t), int(40 * t), int(40 * t))
    # 8-14: Shaded White/Grey for Boing Ball
    for i in range(8, 15):
        t = (i - 7) / 7.0
        pal_over[i] = (int(255 * t), int(255 * t), int(255 * t))
    # 15: Intense specular highlight
    pal_over[15] = (255, 255, 255)

    return pal_base, pal_over


def find_tiles(indexed_img: np.ndarray, n_tiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the 1200 8x8 blocks of an image down to an optimal n_tiles dictionary."""
    tiles = []
    for r in range(ROWS):
        for c in range(COLS):
            patch = indexed_img[r*TILE_H:(r+1)*TILE_H, c*TILE_W:(c+1)*TILE_W].astype(np.float32)
            tiles.append(patch.flatten())
    tiles_arr = np.array(tiles)

    # Fast path if the image naturally has <= n_tiles unique blocks (Guaranteed for the Boing Ball!)
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


def generate_highres(frame_idx: int, fps: int) -> tuple[np.ndarray, np.ndarray]:
    t = frame_idx / max(fps, 1)
    X, Y = np.meshgrid(np.arange(SCREEN_W), np.arange(SCREEN_H))

    img_base = np.zeros((SCREEN_H, SCREEN_W), dtype=np.uint8)
    img_over = np.zeros((SCREEN_H, SCREEN_W), dtype=np.uint8)

    # --- PHYSICS ---
    R = 45
    floor_y = 180
    bounce_height = 100
    bounce_period = 1.5
    
    # Perfect gravity parabola math
    phase = (t % bounce_period) / bounce_period
    bounce_h = 1.0 - 4.0 * (phase - 0.5)**2  # 0.0 at floor, 1.0 at apex
    
    y_pos = floor_y - R - bounce_h * bounce_height
    x_pos = 160 + np.sin(t * 1.8) * 80

    # --- BASE LAYER (Floor Grid + Shadow) ---
    horizon = 100
    mask_floor = Y > horizon
    Y_floor = Y[mask_floor]
    X_floor = X[mask_floor]

    # 3D Perspective projection
    Z = 200.0 / (Y_floor - horizon)
    PX = (X_floor - 160) * Z / 60.0
    PZ = Z - t * 12.0

    # Evaluate grid lines
    line_w = 0.15
    grid = (np.abs(PX % 1.0 - 0.5) < line_w) | (np.abs(PZ % 1.0 - 0.5) < line_w)
    fade = np.clip(1.0 - (Z / 150.0), 0.0, 1.0)
    color_idx = np.clip(np.round(grid * fade * 7), 0, 7).astype(np.uint8)
    img_base[mask_floor] = color_idx

    # Evaluate dynamic shadow
    shadow_x = x_pos
    shadow_y = floor_y + 10  # Slightly offset down for perspective
    dx_shad = (X - shadow_x) / 2.0  # Ellipse is twice as wide as tall
    dy_shad = (Y - shadow_y)
    dist = np.hypot(dx_shad, dy_shad)

    shadow_rad = 35 + bounce_h * 15
    shadow_intensity = np.clip(1.0 - dist / shadow_rad, 0.0, 1.0) * (0.8 - bounce_h * 0.4)
    
    shadow_mask = shadow_intensity > 0.05
    # Shadow overwrites the grid with indices 8-15 (Dark -> Pitch Black)
    shadow_idx = np.clip(8 + np.round(shadow_intensity[shadow_mask] * 7), 8, 15).astype(np.uint8)
    img_base[shadow_mask] = shadow_idx

    # --- OVERLAY LAYER (Boing Ball) ---
    dx_ball = X - x_pos
    dy_ball = Y - y_pos
    r2 = dx_ball**2 + dy_ball**2
    mask_ball = r2 <= R**2

    if np.any(mask_ball):
        dx_b = dx_ball[mask_ball]
        dy_b = dy_ball[mask_ball]
        z_b = np.sqrt(R**2 - dx_b**2 - dy_b**2)

        # Unrotated spherical normal
        nx = dx_b / R
        ny = dy_b / R
        nz = z_b / R

        # Apply 3D Rotation Matrix
        rx = 0.3              # Tilt forward to reveal top
        ry = -t * 3.0         # Spin around axis
        rz = 0.25             # Classic Amiga fixed tilt

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        
        # Matrix multiplication evaluated explicitly for performance
        nx_r = (cy*cz)*nx + (sx*sy*cz - cx*sz)*ny + (cx*sy*cz + sx*sz)*nz
        ny_r = (cy*sz)*nx + (sx*sy*sz + cx*cz)*ny + (cx*sy*sz - sx*cz)*nz
        nz_r = (-sy)*nx + (sx*cy)*ny + (cx*cy)*nz

        # Texture coordinates
        lat = np.arcsin(np.clip(ny_r, -1.0, 1.0))
        lon = np.arctan2(nx_r, nz_r)

        tiles_u = np.floor(lon / (np.pi / 8))  # 16 longitude divisions
        tiles_v = np.floor(lat / (np.pi / 8))  # 8 latitude divisions
        check = (tiles_u + tiles_v) % 2

        # Lighting: Lambertian diffuse + intense specular
        lx, ly, lz = 0.577, -0.577, 0.577  # Light from top-left-front
        dot = nx*lx + ny*ly + nz*lz
        light = np.clip(dot, 0.0, 1.0) * 0.9 + 0.1

        red_color = np.clip(np.round(light * 7), 1, 7).astype(np.uint8)
        white_color = np.clip(np.round(light * 6) + 8, 8, 14).astype(np.uint8)
        color = np.where(check == 0, red_color, white_color)

        # Specular reflection (Phong)
        spec = 2.0 * dot * nz - lz
        color[spec > 0.92] = 15  # Pure white highlight

        img_over[mask_ball] = color

    return img_base, img_over


def build_frame(frame_idx: int, fps: int) -> bytes:
    pal_base, pal_over = make_palettes(frame_idx, fps)
    img_base, img_over = generate_highres(frame_idx, fps)
    
    tiles_base, map_base = find_tiles(img_base, NUM_TILES)
    tiles_over, map_over = find_tiles(img_over, NUM_TILES)

    # Packing order must exactly match the read order in main.c
    return (
        palette_to_bytes(pal_base)                              # [0]     pal1 (Layer 1 / Base)
        + palette_to_bytes(pal_over, transparent_index0=True)   # [32]    pal2 (Layer 2 / Overlay)
        + build_tileset(tiles_over)                             # [64]    tiles2 (Layer 2 / Overlay)
        + build_tileset(tiles_base)                             # [8256]  tiles1 (Layer 1 / Base)
        + bytes(map_over.flatten())                             # [16448] map2 (Layer 2 / Overlay)
        + bytes(map_base.flatten())                             # [17648] map1 (Layer 1 / Base)
    )


def generate_stream(output_path: Path, seconds: float, fps: int) -> None:
    frame_count = int(round(seconds * fps))
    if frame_count <= 0:
        raise ValueError("seconds * fps must produce at least one frame")

    print("Generating V4 Procedural MT62 Showcase - Amiga Boing Ball")
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
    parser = argparse.ArgumentParser(description="Generate V4 Boing Ball procedural MT62 stream")
    parser.add_argument("--output", default="SHOWCASE_V4.BIN", help="Output stream filename")
    parser.add_argument("--seconds", type=float, default=60.0, help="Length of the showcase in seconds")
    parser.add_argument("--fps", type=int, default=24, help="Playback FPS (default: 24)")
    args = parser.parse_args()

    generate_stream(Path(args.output), args.seconds, args.fps)


if __name__ == "__main__":
    main()