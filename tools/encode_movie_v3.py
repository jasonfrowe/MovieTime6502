#!/usr/bin/env python3
"""
encode_movie_v3.py — Striped Dither Encoder for MovieTime6502.

V3 strictly partitions the hardware layers spatially to eliminate residual noise:
1. Image is binned by 2x2 (160x120 target resolution).
2. A 31-color global palette is generated and sorted by brightness.
3. Base Palette (Layer 1) gets the even-indexed colors (16 colors).
4. Overlay Palette (Layer 2) gets the odd-indexed colors (15 colors).
5. Layer 1 draws ONLY the Even columns (0, 2, 4, 6).
6. Layer 2 draws ONLY the Odd columns (1, 3, 5, 7), with Even columns transparent.
"""

import argparse
import os
import struct
import sys
import time

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

SCREEN_W = 320
SCREEN_H = 240
TILE_W = 8
TILE_H = 8
COLS = SCREEN_W // TILE_W
ROWS = SCREEN_H // TILE_H
NUM_TILES = 256
FRAME_BYTES = 18848

HEADER_MAGIC = b"MT62"
HEADER_VERSION = 1


def parse_time(ts: str) -> float:
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
    """Cluster the 8x8 patches of an indexed image down to 256 tiles."""
    tiles = []
    for r in range(ROWS):
        for c in range(COLS):
            patch = indexed_img[r * TILE_H : (r + 1) * TILE_H, c * TILE_W : (c + 1) * TILE_W]
            tiles.append(patch.flatten())
    
    tiles_arr = np.array(tiles, dtype=np.float32)

    # Fast path: Since 50% of columns are locked to 0, we will hit this frequently!
    unique_tiles, inverse_indices = np.unique(tiles_arr, axis=0, return_inverse=True)
    if len(unique_tiles) <= n_tiles:
        tile_dict = np.zeros((n_tiles, TILE_H, TILE_W), dtype=np.uint8)
        tile_dict[:len(unique_tiles)] = unique_tiles.reshape(-1, TILE_H, TILE_W).astype(np.uint8)
        tile_map = inverse_indices.reshape(ROWS, COLS).astype(np.uint8)
    else:
        # Full KMeans fallback if there are more than 256 unique blocks
        km = KMeans(n_clusters=n_tiles, n_init=1, random_state=42, max_iter=30)
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
    # 1. Spatial Binning (reduces to 160x120 target resolution to match the 2-column striping)
    small = cv2.resize(frame_bgr, (SCREEN_W // 2, SCREEN_H // 2), interpolation=cv2.INTER_AREA)
    binned = cv2.resize(small, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_NEAREST)
    frame_rgb = cv2.cvtColor(binned, cv2.COLOR_BGR2RGB)

    # 2. Extract 31-color global palette from the binned image
    pixels = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
    init_centers = prev_centers if (prev_centers is not None and len(prev_centers) == 31) else 'k-means++'
    
    km = KMeans(n_clusters=31, n_init=1, init=init_centers, random_state=42, max_iter=30)
    km.fit(pixels)
    raw_centers = km.cluster_centers_.copy()
    
    # Force empty clusters to black to kill initialization noise
    counts = np.bincount(km.labels_, minlength=31)
    for i in range(31):
        if counts[i] == 0:
            raw_centers[i] = [0, 0, 0]
            
    centers = raw_centers.clip(0, 255).astype(np.uint8)

    # Sort by brightness
    luma = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
    centers = centers[np.argsort(luma)]

    # Split palettes by alternating brightness to guarantee a perfect mix
    base_palette = centers[0::2]  # 16 colors (Even indices)
    overlay_palette = np.zeros((16, 3), dtype=np.uint8)
    overlay_palette[1:] = centers[1::2] # 15 colors (Odd indices), 0 is transparent

    # 3. Base Layer (Even Columns)
    diff_base = frame_rgb.astype(np.int32)[:, :, None, :] - base_palette.astype(np.int32)[None, None, :, :]
    err_base = np.sum(diff_base**2, axis=3)
    base_img_idx = np.argmin(err_base, axis=2).astype(np.uint8)
    base_img_idx[:, 1::2] = 0  # Force Odd columns to 0 to simplify tile clustering

    # 4. Overlay Layer (Odd Columns)
    diff_ov = frame_rgb.astype(np.int32)[:, :, None, :] - overlay_palette.astype(np.int32)[None, None, 1:, :]
    err_ov = np.sum(diff_ov**2, axis=3)
    ov_img_idx = np.argmin(err_ov, axis=2).astype(np.uint8) + 1 # Offset by 1 since 0 is transparent
    ov_img_idx[:, 0::2] = 0    # Force Even columns to transparent (0)

    # 5. Compress the partitioned layers down to 256 tiles each
    base_tiles, base_map = find_tiles(base_img_idx, NUM_TILES)
    overlay_tiles, overlay_map = find_tiles(ov_img_idx, NUM_TILES)

    # 6. Binary Packing
    pal1_bytes   = palette_to_bytes(base_palette)                            # base    -> Layer 1 slot
    pal2_bytes   = palette_to_bytes(overlay_palette, transparent_index0=True)# overlay -> Layer 2 slot
    tiles2_bytes = build_tileset(list(overlay_tiles))                        # overlay -> Layer 2 slot
    tiles1_bytes = build_tileset(list(base_tiles))                           # base    -> Layer 1 slot
    map2_bytes   = bytes(overlay_map.flatten())                              # overlay -> Layer 2 slot
    map1_bytes   = bytes(base_map.flatten())                                 # base    -> Layer 1 slot

    return pal1_bytes + pal2_bytes + tiles2_bytes + tiles1_bytes + map2_bytes + map1_bytes, raw_centers


def reconstruct_frame(frame_bytes: bytes) -> np.ndarray:
    """Decode a packed frame back to an RGB image. Mimics the RP6502 VGA mode 2 hardware."""
    offset = 0
    pal1_data = frame_bytes[offset:offset + 32]; offset += 32
    pal2_data = frame_bytes[offset:offset + 32]; offset += 32
    tiles2_data = frame_bytes[offset:offset + 8192]; offset += 8192
    tiles1_data = frame_bytes[offset:offset + 8192]; offset += 8192
    map2_data  = frame_bytes[offset:offset + 1200]; offset += 1200
    map1_data  = frame_bytes[offset:offset + 1200]

    def parse_pal(data):
        colours = []
        for i in range(16):
            word, = struct.unpack_from("<H", data, i * 2)
            r = (word & 0x1F) << 3
            g = ((word >> 6) & 0x1F) << 3
            b = ((word >> 11) & 0x1F) << 3
            opaque = bool(word & 0x0020)
            colours.append((r, g, b, opaque))
        return colours

    def parse_tiles(data):
        tiles = []
        for t in range(NUM_TILES):
            tile = np.zeros((TILE_H, TILE_W), np.uint8)
            base = t * 32
            for row in range(TILE_H):
                for col in range(TILE_W // 2):
                    byte = data[base + row * 4 + col]
                    tile[row, col * 2]     = byte & 0xF
                    tile[row, col * 2 + 1] = (byte >> 4) & 0xF
            tiles.append(tile)
        return tiles

    pal1 = parse_pal(pal1_data)
    pal2 = parse_pal(pal2_data)
    tiles2 = parse_tiles(tiles2_data)
    tiles1 = parse_tiles(tiles1_data)
    map2 = np.frombuffer(map2_data, np.uint8).reshape(ROWS, COLS)
    map1 = np.frombuffer(map1_data, np.uint8).reshape(ROWS, COLS)

    img = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
    
    # Layer 1 (Base/Middle) - Even columns primarily
    for r in range(ROWS):
        for c in range(COLS):
            tile = tiles1[map1[r, c]]
            for tr in range(TILE_H):
                for tc in range(TILE_W):
                    idx = tile[tr, tc]
                    img[r*TILE_H+tr, c*TILE_W+tc] = pal1[idx][:3]

    # Layer 2 (Overlay/Top) - Odd columns primarily, composites over base using alpha bit
    for r in range(ROWS):
        for c in range(COLS):
            tile = tiles2[map2[r, c]]
            for tr in range(TILE_H):
                for tc in range(TILE_W):
                    idx = tile[tr, tc]
                    if pal2[idx][3]:  # if opaque bit is set
                        img[r*TILE_H+tr, c*TILE_W+tc] = pal2[idx][:3]

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def cmd_encode(args):
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cap = cv2.VideoCapture(args.input)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    start_frame = int((parse_time(args.start) if args.start else 0.0) * source_fps)
    end_frame   = int(parse_time(args.end) * source_fps) if args.end else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_target_frames = int((end_frame - start_frame) / source_fps * args.fps)
    src_step = source_fps / args.fps

    print(f"Encoding {args.input} to {args.output}")
    print(f"Clip: {args.start or '0'} -> {args.end or 'end'}")
    print(f"Targeting {args.fps} FPS, ~{total_target_frames} frames total.\n")

    with open(args.output, "wb") as fout:
        fout.write(struct.pack("<4sBBHHHHI", HEADER_MAGIC, HEADER_VERSION, args.fps, SCREEN_W, SCREEN_H, TILE_W, TILE_H, total_target_frames))
        
        prev_centers = None
        t0 = time.time()
        for i in range(total_target_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + int(i * src_step))
            ret, frame = cap.read()
            if not ret: break
            
            frame_data, prev_centers = encode_frame(resize_with_center_crop(frame, SCREEN_W, SCREEN_H), prev_centers)
            fout.write(frame_data)
            print(f"  [{(i+1)/total_target_frames*100:5.1f}%] frame {i+1}/{total_target_frames}  enc: {(i+1)/(time.time()-t0):.1f} fps", end="\r")
    print(f"\n\nDone! Saved to {args.output}")


def cmd_verify(args):
    os.makedirs(args.debug_dir, exist_ok=True)
    with open(args.stream, "rb") as f:
        header = f.read(18)
        magic, version, fps, w, h, tw, th, frame_count = struct.unpack("<4sBBHHHHI", header)
        print(f"Magic:       {magic}")
        print(f"Version:     {version}")
        print(f"FPS:         {fps}")
        print(f"Resolution:  {w}x{h}")
        print(f"Tile size:   {tw}x{th}")
        print(f"Frames:      {frame_count}")
        
        pil_frames = []
        for i in range(min(args.frames, frame_count)):
            data = f.read(FRAME_BYTES)
            if len(data) < FRAME_BYTES:
                print(f"Truncated at frame {i}")
                break
            recon = reconstruct_frame(data)
            out_path = os.path.join(args.debug_dir, f"frame_{i:04d}.png")
            cv2.imwrite(out_path, recon)
            
            rgb = cv2.cvtColor(recon, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))
            print(f"Saved debug frame {i}")
            
        if pil_frames:
            gif_path = os.path.join(args.debug_dir, "animation.gif")
            pil_frames[0].save(
                gif_path, save_all=True, append_images=pil_frames[1:],
                duration=int(1000 / fps) if fps > 0 else 41, loop=0
            )
            print(f"\nSaved animated GIF to {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="MovieTime6502 V3 (Striped Dither) encoder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="Encode a video clip into MT62 format")
    enc.add_argument("--input", default="Sprites/Metropolis_1927.mp4", help="Source MP4 file")
    enc.add_argument("--start", default="0",  help="Start timestamp HH:MM:SS (default: 0)")
    enc.add_argument("--end",   default=None, help="End timestamp HH:MM:SS (default: end of video)")
    enc.add_argument("--output", default="Movies/MOVIE_metro_v3.BIN", help="Output binary")
    enc.add_argument("--fps", type=int, default=24, help="Target FPS")

    ver = sub.add_parser("verify", help="Decode and inspect a packed stream")
    ver.add_argument("stream", help="MT62 binary stream file")
    ver.add_argument("--debug-dir", default="debug_verify_v3", help="Output dir for PNGs")
    ver.add_argument("--frames", type=int, default=10, help="Number of frames to decode")

    args = parser.parse_args()
    if args.cmd == "encode":
        cmd_encode(args)
    elif args.cmd == "verify":
        cmd_verify(args)

if __name__ == "__main__":
    main()