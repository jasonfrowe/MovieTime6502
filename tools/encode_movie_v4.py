#!/usr/bin/env python3
"""
encode_movie_v4.py - Binned two-layer encoder for the MovieTime6502 demo.

Bins 320x240 -> 160x120 (each binned pixel = 2x2 screen pixels).
Each 8x8 tile is a 4x4 grid of binned cells, drastically limiting unique
tile patterns and making tile clustering stable frame-to-frame.
"""

import argparse
import os
import struct
import sys
import time

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans

SCREEN_W = 320
SCREEN_H = 240
BIN = 2
BIN_W = SCREEN_W // BIN   # 160
BIN_H = SCREEN_H // BIN   # 120
TILE_W = 8
TILE_H = 8
COLS = SCREEN_W // TILE_W  # 40
ROWS = SCREEN_H // TILE_H  # 30
NUM_TILES = 256
FRAME_BYTES = 32 + 32 + 8192 + 8192 + 1200 + 1200  # 18848

HEADER_MAGIC = b"MT62"
HEADER_VERSION = 1


def parse_time(ts):
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


def rgb8_to_rgb555(r, g, b, opaque=True):
    word = ((b >> 3) << 11) | ((g >> 3) << 6) | (r >> 3)
    if opaque:
        word |= 0x0020
    return word


def palette_to_bytes(palette_rgb, transparent_index0=False):
    out = bytearray()
    for idx, (r, g, b) in enumerate(palette_rgb):
        word = rgb8_to_rgb555(int(r), int(g), int(b),
                              opaque=not (transparent_index0 and idx == 0))
        out += struct.pack("<H", word)
    return bytes(out)


def encode_tile(tile_8x8_idx):
    out = bytearray(32)
    for row in range(TILE_H):
        for col in range(TILE_W // 2):
            lo = int(tile_8x8_idx[row, col * 2]) & 0xF
            hi = int(tile_8x8_idx[row, col * 2 + 1]) & 0xF
            out[row * 4 + col] = lo | (hi << 4)
    return bytes(out)


def build_tileset(tiles_idx):
    result = bytearray(NUM_TILES * 32)
    for i, tile in enumerate(tiles_idx):
        result[i * 32 : (i + 1) * 32] = encode_tile(tile)
    return bytes(result)


def resize_with_center_crop(frame_bgr, target_w, target_h):
    src_h, src_w = frame_bgr.shape[:2]
    scale = max(target_w / src_w, target_h / src_h)
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    x = max(0, (scaled_w - target_w) // 2)
    y = max(0, (scaled_h - target_h) // 2)
    return resized[y : y + target_h, x : x + target_w]


def find_tiles(indexed_img, n_tiles):
    """Cluster 8x8 patches into n_tiles representative tiles.

    CRITICAL: tile index 0 is always guaranteed to be the all-zeros tile.
    This prevents KMeans from picking a grain-contaminated tile as the
    black/transparent cluster representative, which causes the repeating
    corner-pixel artifact visible on dark frames.
    """
    tiles = []
    for r in range(ROWS):
        for c in range(COLS):
            patch = indexed_img[r * TILE_H : (r + 1) * TILE_H,
                                c * TILE_W : (c + 1) * TILE_W]
            tiles.append(patch.flatten())

    tiles_arr = np.array(tiles, dtype=np.float32)

    # Fast path: skip KMeans if there are few enough unique patterns.
    unique_tiles, inverse = np.unique(tiles_arr, axis=0, return_inverse=True)
    if len(unique_tiles) <= n_tiles:
        tile_dict = np.zeros((n_tiles, TILE_H, TILE_W), dtype=np.uint8)
        tile_dict[: len(unique_tiles)] = unique_tiles.reshape(
            -1, TILE_H, TILE_W
        ).astype(np.uint8)
        tile_map = inverse.reshape(ROWS, COLS).astype(np.uint8)
    else:
        km = MiniBatchKMeans(n_clusters=n_tiles, n_init=1, random_state=42,
                             max_iter=20)
        tile_ids = km.fit_predict(tiles_arr)
        centres = km.cluster_centers_

        tile_dict = np.zeros((n_tiles, TILE_H, TILE_W), dtype=np.uint8)
        for k in range(n_tiles):
            members = np.where(tile_ids == k)[0]
            if members.size == 0:
                tile_dict[k] = np.clip(
                    np.round(centres[k].reshape(TILE_H, TILE_W)), 0, 15
                ).astype(np.uint8)
            else:
                member_tiles = tiles_arr[members]
                d2 = np.sum((member_tiles - centres[k]) ** 2, axis=1)
                best_idx = members[int(np.argmin(d2))]
                tile_dict[k] = tiles_arr[best_idx].reshape(
                    TILE_H, TILE_W
                ).astype(np.uint8)
        tile_map = tile_ids.reshape(ROWS, COLS).astype(np.uint8)

    # Swap the tile with minimum pixel sum into slot 0, then pin it to zero.
    tile_sums = tile_dict.reshape(n_tiles, -1).sum(axis=1)
    zero_idx = int(np.argmin(tile_sums))
    if zero_idx != 0:
        tile_dict[[0, zero_idx]] = tile_dict[[zero_idx, 0]]
        m0 = tile_map == 0
        mz = tile_map == zero_idx
        tile_map[m0] = zero_idx
        tile_map[mz] = 0
    tile_dict[0] = 0  # pin to truly all-zeros

    return tile_dict, tile_map


def encode_frame(frame_bgr, prev_centers=None):
    # 1. Mild spatial denoise before binning.
    frame_bgr = cv2.bilateralFilter(frame_bgr, d=5, sigmaColor=35, sigmaSpace=35)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 2. Bin to 160x120 via INTER_AREA (averages each 2x2 block, kills grain).
    binned = cv2.resize(frame_rgb, (BIN_W, BIN_H), interpolation=cv2.INTER_AREA)
    pixels = binned.reshape(-1, 3).astype(np.float32)

    # 3. Quantize binned image to 32 colors, warm-started from previous frame.
    init = prev_centers if prev_centers is not None else "k-means++"
    km = KMeans(n_clusters=32, n_init=1, init=init, random_state=42)
    km.fit(pixels)
    raw_centers = km.cluster_centers_.copy()
    centers = raw_centers.clip(0, 255).astype(np.uint8)

    # 4. Stable luma partition: darkest 16 -> base, brightest 15 -> overlay.
    #    Luma is intrinsic to the color values and does not oscillate between
    #    frames the way pixel-count frequency ordering does.
    luma_32 = (0.299 * centers[:, 0] + 0.587 * centers[:, 1]
               + 0.114 * centers[:, 2])
    luma_order = np.argsort(luma_32)
    base_palette = centers[luma_order[:16]]
    overlay_palette = np.zeros((16, 3), dtype=np.uint8)
    overlay_palette[1:] = centers[luma_order[16:31]]

    # 5. For each binned pixel, find the best color across all 31 usable slots.
    #    Slots  0..15 -> base palette indices 0..15
    #    Slots 16..30 -> overlay palette indices 1..15
    all_colors = np.concatenate([base_palette, overlay_palette[1:]], axis=0)
    diff_all = (binned.astype(np.int32)[:, :, None, :]
                - all_colors.astype(np.int32)[None, None, :, :])
    err_all  = np.sum(diff_all ** 2, axis=3)   # (120, 160, 31)
    best_all = np.argmin(err_all, axis=2)       # (120, 160)
    is_base  = best_all < 16

    # 6. Build binned index maps.
    #    Base is always filled (opaque layer); use nearest base color everywhere.
    diff_base = (binned.astype(np.int32)[:, :, None, :]
                 - base_palette.astype(np.int32)[None, None, :, :])
    err_base = np.sum(diff_base ** 2, axis=3)
    base_idx_binned    = np.argmin(err_base, axis=2).astype(np.uint8)
    overlay_idx_binned = np.where(is_base, 0, (best_all - 16 + 1)).astype(np.uint8)

    # 7. Expand 160x120 -> 320x240: each binned pixel becomes a 2x2 block.
    #    Each 8x8 tile is now a 4x4 grid of uniform 2x2 blocks - far fewer
    #    unique tile patterns -> stable tile clustering frame-to-frame.
    base_idx    = np.repeat(np.repeat(base_idx_binned,    BIN, axis=0), BIN, axis=1)
    overlay_idx = np.repeat(np.repeat(overlay_idx_binned, BIN, axis=0), BIN, axis=1)

    # 8. Tile clustering (tile 0 guaranteed all-zeros by find_tiles).
    base_tiles,    base_map    = find_tiles(base_idx,    NUM_TILES)
    overlay_tiles, overlay_map = find_tiles(overlay_idx, NUM_TILES)

    # 9. Force cells that are entirely zero to map 0.
    #    Even if KMeans assigned such a cell to a non-zero cluster, we override
    #    it here so empty regions are guaranteed clean.
    base_cell_max    = base_idx.reshape(   ROWS, TILE_H, COLS, TILE_W).max(axis=(1, 3))
    overlay_cell_max = overlay_idx.reshape(ROWS, TILE_H, COLS, TILE_W).max(axis=(1, 3))
    base_map    = np.where(base_cell_max    == 0, 0, base_map   ).astype(np.uint8)
    overlay_map = np.where(overlay_cell_max == 0, 0, overlay_map).astype(np.uint8)

    # 10. Binary packing (MT62 layout).
    pal1_bytes   = palette_to_bytes(base_palette)
    pal2_bytes   = palette_to_bytes(overlay_palette, transparent_index0=True)
    tiles2_bytes = build_tileset(list(overlay_tiles))
    tiles1_bytes = build_tileset(list(base_tiles))
    map2_bytes   = bytes(overlay_map.flatten())
    map1_bytes   = bytes(base_map.flatten())

    return (pal1_bytes + pal2_bytes + tiles2_bytes + tiles1_bytes
            + map2_bytes + map1_bytes), raw_centers


def reconstruct_frame(frame_data):
    """Decode one MT62 frame to RGB for debugging."""
    off = 0
    pal1_raw    = frame_data[off:off+32];   off += 32
    pal2_raw    = frame_data[off:off+32];   off += 32
    tiles2_raw  = frame_data[off:off+8192]; off += 8192
    tiles1_raw  = frame_data[off:off+8192]; off += 8192
    map2_raw    = frame_data[off:off+1200]; off += 1200
    map1_raw    = frame_data[off:off+1200]

    def parse_pal(raw):
        out = []
        for i in range(16):
            word, = struct.unpack_from("<H", raw, i * 2)
            r = (word & 0x1F) << 3
            g = ((word >> 6) & 0x1F) << 3
            b = ((word >> 11) & 0x1F) << 3
            out.append((r, g, b, bool(word & 0x0020)))
        return out

    def parse_tiles(raw):
        tiles = []
        for t in range(NUM_TILES):
            tile = np.zeros((TILE_H, TILE_W), np.uint8)
            base_off = t * 32
            for row in range(TILE_H):
                for col in range(TILE_W // 2):
                    byte = raw[base_off + row * 4 + col]
                    tile[row, col * 2]     = byte & 0xF
                    tile[row, col * 2 + 1] = (byte >> 4) & 0xF
            tiles.append(tile)
        return tiles

    pal1   = parse_pal(pal1_raw)
    pal2   = parse_pal(pal2_raw)
    tiles1 = parse_tiles(tiles1_raw)
    tiles2 = parse_tiles(tiles2_raw)
    map1 = np.frombuffer(map1_raw, np.uint8).reshape(ROWS, COLS)
    map2 = np.frombuffer(map2_raw, np.uint8).reshape(ROWS, COLS)

    img = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            tile = tiles1[map1[r, c]]
            for tr in range(TILE_H):
                for tc in range(TILE_W):
                    img[r*TILE_H+tr, c*TILE_W+tc] = pal1[tile[tr, tc]][:3]
    for r in range(ROWS):
        for c in range(COLS):
            tile = tiles2[map2[r, c]]
            for tr in range(TILE_H):
                for tc in range(TILE_W):
                    idx = tile[tr, tc]
                    if pal2[idx][3]:
                        img[r*TILE_H+tr, c*TILE_W+tc] = pal2[idx][:3]
    return img


def cmd_encode(args):
    start_sec = parse_time(args.start)
    end_sec   = parse_time(args.end)
    duration  = end_sec - start_sec
    total_target_frames = int(duration * args.fps)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open {args.input}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    start_src_frame = max(0, int(round(start_sec * source_fps)))
    src_frame_step  = source_fps / float(args.fps)

    print(f"Encoding {args.input} -> {args.output}")
    print(f"Clip: {args.start} -> {args.end} ({duration:.1f}s)")
    print(f"Targeting {args.fps} FPS, ~{total_target_frames} frames total.\n")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "wb") as fout:
        header = struct.pack(
            "<4sBBHHHHI",
            HEADER_MAGIC, HEADER_VERSION, args.fps,
            SCREEN_W, SCREEN_H, TILE_W, TILE_H,
            total_target_frames,
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
            print(f"  [{pct:5.1f}%] frame {frame_idx+1}/{total_target_frames}"
                  f"  enc: {fps_enc:.1f} fps", end="\r")

    cap.release()
    print(f"\n\nDone! Saved to {args.output}")


def cmd_verify(args):
    os.makedirs(args.debug_dir, exist_ok=True)
    with open(args.stream, "rb") as f:
        hdr = f.read(18)
        magic, version, fps, w, h, tw, th, frame_count = struct.unpack("<4sBBHHHHI", hdr)
        print(f"Magic: {magic}  Version: {version}  FPS: {fps}")
        print(f"Resolution: {w}x{h}  Tile: {tw}x{th}  Frames: {frame_count}")
        pil_frames = []
        for i in range(min(args.frames, frame_count)):
            data = f.read(FRAME_BYTES)
            if len(data) < FRAME_BYTES:
                print(f"Truncated at frame {i}")
                break
            img = reconstruct_frame(data)
            out_path = os.path.join(args.debug_dir, f"frame_{i:04d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            pil_frames.append(Image.fromarray(img))
            print(f"  Saved {out_path}")
            
        if pil_frames:
            gif_path = os.path.join(args.debug_dir, "animation.gif")
            pil_frames[0].save(
                gif_path, save_all=True, append_images=pil_frames[1:],
                duration=int(1000 / fps) if fps > 0 else 41, loop=0
            )
            print(f"\nSaved animated GIF to {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="MovieTime6502 V4 encoder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="Encode a video clip into MT62 format")
    enc.add_argument("--input", default="Sprites/Metropolis_1927.mp4")
    enc.add_argument("--start", required=True, help="Start timestamp HH:MM:SS")
    enc.add_argument("--end",   required=True, help="End timestamp HH:MM:SS")
    enc.add_argument("--output", default="Movies/MOVIE_metro.BIN")
    enc.add_argument("--fps", type=int, default=24)

    ver = sub.add_parser("verify", help="Decode frames to PNG for inspection")
    ver.add_argument("stream", help="MT62 binary stream file")
    ver.add_argument("--debug-dir", default="debug_verify_v4")
    ver.add_argument("--frames", type=int, default=10)

    args = parser.parse_args()
    if args.cmd == "encode":
        cmd_encode(args)
    elif args.cmd == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
