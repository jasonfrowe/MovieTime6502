#!/usr/bin/env python3
"""
encode_movie_v2.py — Simplified Offline encoder for the MovieTime6502 demo.

A fresh start implementing a 32-color separation method:
1. Quantize full frame to 32 colors using KMeans.
2. Top 16 colors -> Base Palette (Layer 1).
3. Next 15 colors -> Overlay Palette (Layer 2) (Index 0 is transparent).
4. Base Map -> Fit frame to Base Palette.
5. Overlay Map -> Fit error residual to Overlay Palette.
6. Pack using standard MT62 layout.
"""

import argparse
import os
import struct
import sys
import time

import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

SCREEN_W = 320
SCREEN_H = 240
TILE_W = 8
TILE_H = 8
COLS = SCREEN_W // TILE_W
ROWS = SCREEN_H // TILE_H
NUM_TILES = 256
COLOURS = 16

HEADER_MAGIC = b"MT62"
HEADER_VERSION = 1


def parse_time(ts: str) -> float:
    """Parse HH:MM:SS or MM:SS into seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


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


def build_tileset(tiles_idx: list[np.ndarray]) -> bytes:
    result = bytearray(NUM_TILES * 32)
    for i, tile in enumerate(tiles_idx):
        result[i * 32 : (i + 1) * 32] = encode_tile(tile)
    return bytes(result)


def resize_with_center_crop(frame_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    src_h, src_w = frame_bgr.shape[:2]
    scale = max(target_w / src_w, target_h / src_h)
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))

    resized = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    x = max(0, (scaled_w - target_w) // 2)
    y = max(0, (scaled_h - target_h) // 2)
    return resized[y : y + target_h, x : x + target_w]


def find_tiles(indexed_img: np.ndarray, n_tiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the 8x8 patches of an indexed image to find the optimal tile dictionary."""
    tiles = []
    for r in range(ROWS):
        for c in range(COLS):
            patch = indexed_img[r * TILE_H : (r + 1) * TILE_H, c * TILE_W : (c + 1) * TILE_W]
            tiles.append(patch.flatten())
    
    tiles_arr = np.array(tiles, dtype=np.float32)

    # MiniBatch is much faster for the 1200 tile blocks than standard KMeans
    km = MiniBatchKMeans(n_clusters=n_tiles, n_init=1, random_state=42, max_iter=10)
    tile_ids = km.fit_predict(tiles_arr)
    centres = km.cluster_centers_

    tile_dict = np.zeros((n_tiles, TILE_H, TILE_W), dtype=np.uint8)
    for k in range(n_tiles):
        members = np.where(tile_ids == k)[0]
        if members.size == 0:
            tile_dict[k] = np.clip(np.round(centres[k].reshape(TILE_H, TILE_W)), 0, 15).astype(np.uint8)
        else:
            member_tiles = tiles_arr[members]
            d2 = np.sum((member_tiles - centres[k]) ** 2, axis=1)
            best_idx = members[int(np.argmin(d2))]
            tile_dict[k] = tiles_arr[best_idx].reshape(TILE_H, TILE_W).astype(np.uint8)

    tile_map = tile_ids.reshape(ROWS, COLS).astype(np.uint8)
    return tile_dict, tile_map


def encode_frame(frame_bgr: np.ndarray, prev_centers=None) -> tuple[bytes, np.ndarray]:
    # 1. Spatial Denoise (reduces film grain, banding, and eases quantization)
    frame_bgr = cv2.bilateralFilter(frame_bgr, d=5, sigmaColor=35, sigmaSpace=35)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pixels = frame_rgb.reshape(-1, 3).astype(np.float32)

    # 2. Color Quantization with Temporal Stability (reduces frame-to-frame flicker)
    init_centers = prev_centers if prev_centers is not None else 'k-means++'
    km = KMeans(n_clusters=32, n_init=1, init=init_centers, random_state=42)
    labels = km.fit_predict(pixels)
    raw_centers = km.cluster_centers_
    centers = raw_centers.clip(0, 255).astype(np.uint8)

    # Sort by frequency so the most common colors form our base layer
    counts = np.bincount(labels, minlength=32)
    order = np.argsort(-counts)
    
    base_palette = centers[order[:16]]
    overlay_palette = np.zeros((16, 3), dtype=np.uint8)
    overlay_palette[1:] = centers[order[16:31]]  # Index 0 is transparent

    # 2. Base layer mapping (Layer 1)
    diff_base = frame_rgb.astype(np.int32)[:, :, None, :] - base_palette.astype(np.int32)[None, None, :, :]
    err_base = np.sum(diff_base**2, axis=3)
    base_img_idx = np.argmin(err_base, axis=2).astype(np.uint8)
    best_err_base = np.take_along_axis(err_base, base_img_idx[:, :, None], axis=2)[:, :, 0]

    base_tiles, base_map = find_tiles(base_img_idx, NUM_TILES)

    # 3. Residual Generation for Layer 2
    diff_ov = frame_rgb.astype(np.int32)[:, :, None, :] - overlay_palette.astype(np.int32)[None, None, 1:, :]
    err_ov = np.sum(diff_ov**2, axis=3)
    ov_img_idx_raw = np.argmin(err_ov, axis=2).astype(np.uint8)
    best_err_ov = np.take_along_axis(err_ov, ov_img_idx_raw[:, :, None], axis=2)[:, :, 0]

    # Mask: use overlay where it visually improves the error residual
    use_overlay = (best_err_base > 100) & (best_err_ov < best_err_base)

    # 3. Clean up the overlay mask to prevent 1-pixel "sparkle" noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    use_overlay_clean = cv2.morphologyEx((use_overlay.astype(np.uint8) * 255), cv2.MORPH_OPEN, kernel)
    overlay_img_idx = np.where(use_overlay_clean > 0, ov_img_idx_raw + 1, 0).astype(np.uint8)

    overlay_tiles, overlay_map = find_tiles(overlay_img_idx, NUM_TILES)

    # 4. Binary Packing (matching standard MovieTime6502 packing format exactly)
    pal1_bytes   = palette_to_bytes(base_palette)                            # base    -> Layer 1 slot
    pal2_bytes   = palette_to_bytes(overlay_palette, transparent_index0=True)# overlay -> Layer 2 slot
    tiles2_bytes = build_tileset(list(overlay_tiles))                        # overlay -> Layer 2 slot
    tiles1_bytes = build_tileset(list(base_tiles))                           # base    -> Layer 1 slot
    map2_bytes   = bytes(overlay_map.flatten())                              # overlay -> Layer 2 slot
    map1_bytes   = bytes(base_map.flatten())                                 # base    -> Layer 1 slot

    return pal1_bytes + pal2_bytes + tiles2_bytes + tiles1_bytes + map2_bytes + map1_bytes, raw_centers


def main():
    parser = argparse.ArgumentParser(description="MovieTime6502 V2 encoder")
    parser.add_argument("--input", default="Sprites/Metropolis_1927.mp4", help="Source MP4 file")
    parser.add_argument("--start", required=True, help="Start timestamp HH:MM:SS")
    parser.add_argument("--end", required=True, help="End timestamp HH:MM:SS")
    parser.add_argument("--output", default="Movies/MOVIE_metro.BIN", help="Output binary")
    parser.add_argument("--fps", type=int, default=24, help="Target FPS")

    args = parser.parse_args()

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)
    duration = end_sec - start_sec
    total_target_frames = int(duration * args.fps)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open {args.input}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    start_src_frame = max(0, int(round(start_sec * source_fps)))
    src_frame_step = source_fps / float(args.fps)

    print(f"Encoding {args.input} to {args.output}")
    print(f"Clip: {args.start} -> {args.end} ({duration:.1f}s)")
    print(f"Targeting {args.fps} FPS, ~{total_target_frames} frames total.\n")

    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "wb") as fout:
        # Write Header
        header = struct.pack(
            "<4sBBHHHHI",
            HEADER_MAGIC, HEADER_VERSION, args.fps,
            SCREEN_W, SCREEN_H, TILE_W, TILE_H,
            total_target_frames
        )
        fout.write(header)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_src_frame)
        current_src_frame = start_src_frame - 1
        current_bgr = None
        next_src_frame = start_src_frame
        prev_centers = None

        t0 = time.time()

        for frame_idx in range(total_target_frames):
            target_src_frame = start_src_frame + int(round(frame_idx * src_frame_step))

            while current_src_frame < target_src_frame:
                ret, bgr = cap.read()
                if not ret:
                    break
                current_bgr = bgr
                current_src_frame = next_src_frame
                next_src_frame += 1

            if current_bgr is None:
                break

            bgr_cropped = resize_with_center_crop(current_bgr, SCREEN_W, SCREEN_H)
            frame_data, prev_centers = encode_frame(bgr_cropped, prev_centers)
            fout.write(frame_data)

            elapsed = time.time() - t0
            fps_enc = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            pct = (frame_idx + 1) / total_target_frames * 100
            print(f"  [{pct:5.1f}%] frame {frame_idx+1}/{total_target_frames}  enc: {fps_enc:.1f} fps", end="\r")

    print(f"\n\nDone! Saved to {args.output}")


if __name__ == "__main__":
    main()