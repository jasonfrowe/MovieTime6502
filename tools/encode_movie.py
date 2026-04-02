#!/usr/bin/env python3
"""
encode_movie.py — Offline encoder for the MovieTime6502 demo.

Converts a video clip into a flat binary stream that the RP6502 player reads
frame-by-frame from USB mass storage.

File format
-----------
Header (18 bytes, all little-endian):
    4 bytes  magic      b"MT62"
    1 byte   version    1
    1 byte   fps        nominal frames per second (e.g. 24)
    2 bytes  width      pixels (320)
    2 bytes  height     pixels (240)
    2 bytes  tile_w     tile width  (8)
    2 bytes  tile_h     tile height (8)
    4 bytes  frame_count

Each frame (18,848 bytes, fixed size):
    32 bytes  palette1   16 × RGB555 little-endian (layer 1 overlay)
    32 bytes  palette2   16 × RGB555 little-endian (layer 2 base)
    8192 bytes tiles2    256 tiles × 32 bytes (8×8 4-bit), base layer
    8192 bytes tiles1    256 tiles × 32 bytes (8×8 4-bit), overlay layer
    1200 bytes map2      40×30 tile-IDs, base layer
    1200 bytes map1      40×30 tile-IDs, overlay layer

RGB555 encoding used by the RP6502 VGA:
    #define COLOR_FROM_RGB8(r,g,b) (((b>>3)<<11)|((g>>3)<<6)|(r>>3))
    Alpha/opaque bit is bit5 of the high byte — set for all visible colours.

Usage
-----
    python tools/encode_movie.py encode \
        --input  Metropolis_1927.mp4 \
        --start  "01:23:00" \
        --end    "01:26:00" \
        --output MOVIE.BIN \
        [--fps 24] \
        [--debug-dir debug/]

Dependencies: opencv-python, numpy, scikit-learn, Pillow
    pip install opencv-python numpy scikit-learn Pillow
"""

import argparse
import os
import struct
import sys
import time

import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from PIL import Image

# ---------------------------------------------------------------------------
# Constants that must match constants.h and the VGA mode-2 config
# ---------------------------------------------------------------------------
SCREEN_W   = 320
SCREEN_H   = 240
TILE_W     = 8
TILE_H     = 8
COLS       = SCREEN_W // TILE_W     # 40
ROWS       = SCREEN_H // TILE_H     # 30
NUM_TILES  = 256
COLOURS    = 16                     # per palette (4-bit)
TOTAL_COLOURS = 32                  # across both palettes
VISIBLE_OVERLAY_COLOURS = COLOURS - 1  # palette index 0 is reserved as transparent

PALETTE_BYTES  = 2 * COLOURS        # 32
TILES_BYTES    = NUM_TILES * TILE_H * (TILE_W // 2)  # 256*8*4 = 8192
MAP_BYTES      = COLS * ROWS        # 1200
FRAME_BYTES    = (PALETTE_BYTES * 2) + (TILES_BYTES * 2) + (MAP_BYTES * 2)  # 18 848

HEADER_MAGIC   = b"MT62"
HEADER_VERSION = 1
HEADER_BYTES   = struct.calcsize("<4sBBHHHHI")

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def rgb8_to_rgb555(r: int, g: int, b: int, opaque: bool = True) -> int:
    """Pack an 8-bit RGB triplet into RP6502 RGB555 with optional alpha."""
    # COLOR_FROM_RGB8(r,g,b) = (((b>>3)<<11)|((g>>3)<<6)|(r>>3))
    # Opaque bit = bit 5 of byte 1, i.e. 0x0020 ORed in
    word = ((b >> 3) << 11) | ((g >> 3) << 6) | (r >> 3)
    if opaque:
        word |= 0x0020
    return word


def palette_to_bytes(palette_rgb: np.ndarray, transparent_index0: bool = False) -> bytes:
    """Convert an Nx3 uint8 array to 2*N bytes of RP6502 RGB555."""
    out = bytearray()
    for idx, (r, g, b) in enumerate(palette_rgb):
        word = rgb8_to_rgb555(int(r), int(g), int(b), opaque=not (transparent_index0 and idx == 0))
        out += struct.pack("<H", word)
    return bytes(out)


def resize_with_center_crop(frame_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize to fill the target, then center-crop any excess."""
    src_h, src_w = frame_bgr.shape[:2]
    scale = max(target_w / src_w, target_h / src_h)
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))

    resized = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    x = max(0, (scaled_w - target_w) // 2)
    y = max(0, (scaled_h - target_h) // 2)
    return resized[y:y + target_h, x:x + target_w]


# ---------------------------------------------------------------------------
# Tile encoding
# ---------------------------------------------------------------------------

def encode_tile(tile_8x8_idx: np.ndarray) -> bytes:
    """
    Encode an 8×8 array of 4-bit palette indices (0-15) in 'tall' bitmap
    format expected by VGA mode 2:
        for each row (8 rows):
            col[0..3] — 4 bytes, each 2 pixels packed as low_nibble|high_nibble<<4
    Result is 32 bytes.
    """
    out = bytearray(32)
    for row in range(TILE_H):
        for col in range(TILE_W // 2):
            lo = int(tile_8x8_idx[row, col * 2]) & 0xF
            hi = int(tile_8x8_idx[row, col * 2 + 1]) & 0xF
            out[row * 4 + col] = lo | (hi << 4)
    return bytes(out)


def build_tileset(tiles_idx: list[np.ndarray]) -> bytes:
    """Encode a list of up to 256 8×8 index arrays into 8192 bytes."""
    assert len(tiles_idx) <= NUM_TILES
    result = bytearray(TILES_BYTES)
    for i, tile in enumerate(tiles_idx):
        encoded = encode_tile(tile)
        result[i * 32: (i + 1) * 32] = encoded
    return bytes(result)


# ---------------------------------------------------------------------------
# Frame encoder
# ---------------------------------------------------------------------------

def quantize_frame(frame_rgb: np.ndarray, n_colors: int, palette_hint=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a 320×240 RGB frame to n_colors using K-Means.
    Returns (palette_rgb [n_colors, 3], indexed_image [240, 320]).
    palette_hint: (n_colors, 3) warm-start centres to speed up convergence.
    """
    h, w, _ = frame_rgb.shape
    pixels = frame_rgb.reshape(-1, 3).astype(np.float32)
    init = palette_hint.astype(np.float32) if palette_hint is not None else "k-means++"
    km = MiniBatchKMeans(n_clusters=n_colors, init=init, n_init=1,
                         max_iter=100, random_state=42, batch_size=4096)
    km.fit(pixels)
    labels = km.predict(pixels).reshape(h, w)
    centres = km.cluster_centers_.clip(0, 255).astype(np.uint8)
    return centres, labels


def find_tiles(indexed_img: np.ndarray, n_tiles: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster the 1200 8×8 tiles of an indexed image into n_tiles representative
    tiles.  Returns (tile_dict [n_tiles, 8, 8], map [ROWS, COLS]).
    """
    tiles = []
    for r in range(ROWS):
        for c in range(COLS):
            patch = indexed_img[r*TILE_H:(r+1)*TILE_H, c*TILE_W:(c+1)*TILE_W].astype(np.float32)
            tiles.append(patch.flatten())
    tiles_arr = np.array(tiles)  # (1200, 64)

    km = MiniBatchKMeans(n_clusters=n_tiles, n_init=3, max_iter=100,
                         random_state=42, batch_size=512)
    km.fit(tiles_arr)
    tile_ids = km.predict(tiles_arr)
    centres = km.cluster_centers_.clip(0, 15).round().astype(np.uint8)
    tile_dict = centres.reshape(n_tiles, TILE_H, TILE_W)
    tile_map = tile_ids.reshape(ROWS, COLS).astype(np.uint8)
    return tile_dict, tile_map


def encode_frame(frame_bgr: np.ndarray, prev_palette1=None, prev_palette2=None) -> bytes:
    """
    Encode a single 320×240 BGR frame into FRAME_BYTES bytes.
    Strategy:
      1. Global 32-colour quantization → split into two 16-colour palettes.
      2. Layer 2 (base): 256-tile dictionary from full quantized image.
      3. Layer 1 (overlay): 256-tile dictionary from residual / correction image.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ----- Step 1: 32-colour global palette --------------------------------
    pal32, labels32 = quantize_frame(frame_rgb, TOTAL_COLOURS, palette_hint=None)

    # Split by luminance. Base gets 16 visible colours. Overlay gets 15 visible
    # colours plus palette index 0 reserved as transparent.
    luma = 0.299 * pal32[:, 0] + 0.587 * pal32[:, 1] + 0.114 * pal32[:, 2]
    order = np.argsort(luma)          # darkest → brightest
    pal2_global_idx = order[:COLOURS]                   # 16 darker colours → base layer
    pal1_global_idx = order[-VISIBLE_OVERLAY_COLOURS:]  # 15 brighter colours → overlay layer

    palette2_rgb = pal32[pal2_global_idx]   # (16, 3)
    palette1_rgb = np.zeros((COLOURS, 3), dtype=np.uint8)
    palette1_rgb[1:] = pal32[pal1_global_idx]

    # Remap labels32 → two local index spaces
    remap2 = np.full(TOTAL_COLOURS, 0, np.uint8)
    remap1 = np.full(TOTAL_COLOURS, 0, np.uint8)
    for loc, glob in enumerate(pal2_global_idx):
        remap2[glob] = loc
    for loc, glob in enumerate(pal1_global_idx, start=1):
        remap1[glob] = loc

    img2_idx = remap2[labels32]   # image quantized to palette2 indices (0-15)
    img1_idx = remap1[labels32]   # image quantized to palette1 indices (0-15)

    # ----- Step 2: Base layer tile dictionary (256 tiles) -------------------
    tile_dict2, tile_map2 = find_tiles(img2_idx, NUM_TILES)

    # ----- Step 3: Overlay layer tile dictionary (256 tiles) ----------------
    # Use the same per-pixel quantization but remapped to palette1 indices.
    tile_dict1, tile_map1 = find_tiles(img1_idx, NUM_TILES)

    # ----- Pack the frame ---------------------------------------------------
    pal1_bytes = palette_to_bytes(palette1_rgb, transparent_index0=True)   # 32 bytes
    pal2_bytes = palette_to_bytes(palette2_rgb)   # 32 bytes
    tiles2_bytes = build_tileset(list(tile_dict2))  # 8192 bytes
    tiles1_bytes = build_tileset(list(tile_dict1))  # 8192 bytes
    map2_bytes = bytes(tile_map2.flatten())       # 1200 bytes
    map1_bytes = bytes(tile_map1.flatten())       # 1200 bytes

    frame_bytes = pal1_bytes + pal2_bytes + tiles2_bytes + tiles1_bytes + map2_bytes + map1_bytes
    assert len(frame_bytes) == FRAME_BYTES, f"Frame size mismatch: {len(frame_bytes)}"
    return frame_bytes


# ---------------------------------------------------------------------------
# Debug / verification helpers
# ---------------------------------------------------------------------------

def reconstruct_frame(frame_bytes: bytes) -> np.ndarray:
    """
    Decode a packed frame back to an RGB image. Used for PC-side verification.
    Returns an SCREEN_H × SCREEN_W × 3 uint8 array.
    """
    offset = 0
    pal1_data = frame_bytes[offset:offset + PALETTE_BYTES]; offset += PALETTE_BYTES
    pal2_data = frame_bytes[offset:offset + PALETTE_BYTES]; offset += PALETTE_BYTES
    tiles2_data = frame_bytes[offset:offset + TILES_BYTES]; offset += TILES_BYTES
    tiles1_data = frame_bytes[offset:offset + TILES_BYTES]; offset += TILES_BYTES
    map2_data  = frame_bytes[offset:offset + MAP_BYTES];    offset += MAP_BYTES
    map1_data  = frame_bytes[offset:offset + MAP_BYTES]

    def parse_pal(data):
        colours = []
        for i in range(COLOURS):
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

    # Layer 2 = base
    img = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            tile = tiles2[map2[r, c]]
            for tr in range(TILE_H):
                for tc in range(TILE_W):
                    idx = tile[tr, tc]
                    rgb = pal2[idx][:3]
                    img[r*TILE_H+tr, c*TILE_W+tc] = rgb

    # Layer 1 = overlay (composited over base — non-zero index replaces base)
    for r in range(ROWS):
        for c in range(COLS):
            tile = tiles1[map1[r, c]]
            for tr in range(TILE_H):
                for tc in range(TILE_W):
                    idx = tile[tr, tc]
                    if pal1[idx][3]:
                        rgb = pal1[idx][:3]
                        img[r*TILE_H+tr, c*TILE_W+tc] = rgb

    return img


def save_debug_frame(frame_idx: int, frame_bytes: bytes, debug_dir: str):
    """Save palette visualization and reconstructed frame PNG for inspection."""
    os.makedirs(debug_dir, exist_ok=True)
    recon = reconstruct_frame(frame_bytes)
    img = Image.fromarray(recon, "RGB")
    img.save(os.path.join(debug_dir, f"frame_{frame_idx:04d}.png"))


# ---------------------------------------------------------------------------
# Time string helpers
# ---------------------------------------------------------------------------

def parse_time(ts: str) -> float:
    """Parse HH:MM:SS or MM:SS into seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


# ---------------------------------------------------------------------------
# Main encode loop
# ---------------------------------------------------------------------------

def encode(input_path: str, start_ts: str, end_ts: str, output_path: str,
           fps: int = 24, debug_dir: str = None, debug_every: int = 24):

    start_sec = parse_time(start_ts)
    end_sec   = parse_time(end_ts)
    duration  = end_sec - start_sec

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    print(f"Source: {input_path}")
    print(f"Source FPS: {source_fps:.3f}")
    print(f"Clip: {start_ts} → {end_ts}  ({duration:.1f}s)")
    print(f"Target FPS: {fps}")

    # Compute which source frames to sample
    total_frames = int(duration * fps)
    source_times = [start_sec + i / fps for i in range(total_frames)]

    print(f"Total encoded frames: {total_frames}")
    print(f"Frame payload: {FRAME_BYTES:,} bytes")
    print(f"Total stream size: {total_frames * FRAME_BYTES / 1024 / 1024:.1f} MiB")
    print(f"Required read bandwidth: {total_frames * FRAME_BYTES / duration / 1024:.0f} KiB/s "
          f"(measured ceiling ~515 KiB/s)\n")

    with open(output_path, "wb") as fout:
        # Write header
        header = struct.pack("<4sBBHHHHI",
            HEADER_MAGIC, HEADER_VERSION, fps,
            SCREEN_W, SCREEN_H, TILE_W, TILE_H,
            total_frames)
        fout.write(header)

        total_written = 0
        t0 = time.time()

        for frame_idx, pts in enumerate(source_times):
            cap.set(cv2.CAP_PROP_POS_MSEC, pts * 1000.0)
            ok, bgr = cap.read()
            if not ok:
                print(f"WARNING: Could not read frame {frame_idx} at {pts:.3f}s — repeating last")
                # Re-use the last encoded frame bytes
                fout.write(frame_data)
                total_written += FRAME_BYTES
                continue

            # Resize to fill 320x240, cropping left/right as needed.
            bgr = resize_with_center_crop(bgr, SCREEN_W, SCREEN_H)

            frame_data = encode_frame(bgr)
            fout.write(frame_data)
            total_written += FRAME_BYTES

            if debug_dir and (frame_idx % debug_every == 0):
                save_debug_frame(frame_idx, frame_data, debug_dir)

            # Progress
            if frame_idx % fps == 0 or frame_idx == total_frames - 1:
                elapsed = time.time() - t0
                fps_enc = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                pct = (frame_idx + 1) / total_frames * 100
                print(f"  [{pct:5.1f}%] frame {frame_idx+1}/{total_frames}  "
                      f"encode {fps_enc:.1f} fr/s  "
                      f"written {total_written/1024:.0f} KiB", end="\r")

    cap.release()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {output_path}  ({os.path.getsize(output_path)/1024/1024:.1f} MiB)")
    print(f"\nCopy MOVIE.BIN to the root of your USB drive, then launch the ROM.")


# ---------------------------------------------------------------------------
# Verify subcommand
# ---------------------------------------------------------------------------

def verify(stream_path: str, debug_dir: str, n_frames: int = 10):
    """Decode the first n_frames from a packed stream and save PNGs for inspection."""
    os.makedirs(debug_dir, exist_ok=True)
    with open(stream_path, "rb") as f:
        header = f.read(HEADER_BYTES)
        magic, version, fps, w, h, tw, th, frame_count = struct.unpack("<4sBBHHHHI", header)
        print(f"Magic:       {magic}")
        print(f"Version:     {version}")
        print(f"FPS:         {fps}")
        print(f"Resolution:  {w}×{h}")
        print(f"Tile size:   {tw}×{th}")
        print(f"Frames:      {frame_count}")
        print(f"Stream size: {frame_count * FRAME_BYTES / 1024 / 1024:.1f} MiB\n")
        for i in range(min(n_frames, frame_count)):
            data = f.read(FRAME_BYTES)
            if len(data) < FRAME_BYTES:
                print(f"Truncated at frame {i}"); break
            save_debug_frame(i, data, debug_dir)
            print(f"Saved debug frame {i}")
    print(f"\nFrames written to {debug_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MovieTime6502 offline encoder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="Encode a video clip into MT62 format")
    enc.add_argument("--input",  required=True, help="Source MP4 file")
    enc.add_argument("--start",  required=True, help="Start timestamp HH:MM:SS")
    enc.add_argument("--end",    required=True, help="End timestamp HH:MM:SS")
    enc.add_argument("--output", default="MOVIE.BIN", help="Output binary (default: MOVIE.BIN)")
    enc.add_argument("--fps",    type=int, default=24, help="Target playback FPS (default: 24)")
    enc.add_argument("--debug-dir", default=None, help="Save debug PNGs here")
    enc.add_argument("--debug-every", type=int, default=24,
                     help="Save a debug PNG every N frames (default: 24)")

    ver = sub.add_parser("verify", help="Decode and inspect a packed stream")
    ver.add_argument("stream", help="MT62 binary stream file")
    ver.add_argument("--debug-dir", default="debug_verify", help="Output dir for PNGs")
    ver.add_argument("--frames", type=int, default=10, help="Number of frames to decode")

    args = parser.parse_args()

    if args.cmd == "encode":
        encode(args.input, args.start, args.end, args.output,
               fps=args.fps, debug_dir=args.debug_dir, debug_every=args.debug_every)
    elif args.cmd == "verify":
        verify(args.stream, args.debug_dir, args.frames)


if __name__ == "__main__":
    main()
