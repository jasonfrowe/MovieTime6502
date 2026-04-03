#!/usr/bin/env python3
"""
encode_movie_lb.py — Low-Bandwidth Encoder for MovieTime6502.

ML62 format: mathematically fixed tile dictionaries stream only palette + map,
achieving an ~8× bandwidth reduction vs MT62 (2,464 vs 18,848 bytes/frame).

Fixed tile scheme (256 tiles, never changes across frames):
  Tile N: H = N >> 4  — colour index for the top 4 screen rows of the tile
           L = N & 0xF — colour index for the bottom 4 screen rows of the tile
  tiles1.bin (base layer):    active on even screen cols (0, 2, 4, 6); odd = 0
  tiles2.bin (overlay layer): active on odd  screen cols (1, 3, 5, 7); even = 0

Per-frame layout (2,464 bytes):
  [0]       32 bytes   palette1  (base layer,    16 × RGB555)
  [32]      32 bytes   palette2  (overlay layer, 16 × RGB555, index 0 transparent)
  [64]    1200 bytes   map1      (40×30 tile IDs, base layer)
  [1264]  1200 bytes   map2      (40×30 tile IDs, overlay layer)

File header (18 bytes, identical to MT62):
  4  bytes  magic   "ML62"
  1  byte   version  1
  1  byte   fps
  2  bytes  width    320
  2  bytes  height   240
  2  bytes  tile_w   8
  2  bytes  tile_h   8
  4  bytes  frame_count (little-endian uint32)
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
COLS = SCREEN_W // TILE_W   # 40
ROWS = SCREEN_H // TILE_H   # 30

LB_FRAME_BYTES = 2464    # 32 + 32 + 1200 + 1200
HEADER_MAGIC = b"ML62"
HEADER_VERSION = 1


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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
        word = rgb8_to_rgb555(int(r), int(g), int(b),
                               opaque=not (transparent_index0 and idx == 0))
        out += struct.pack("<H", word)
    return bytes(out)


def parse_rgb555_palette(data: bytes) -> list[tuple[int, int, int, bool]]:
    colours = []
    for i in range(16):
        word, = struct.unpack_from("<H", data, i * 2)
        r = (word & 0x1F) << 3
        g = ((word >> 6) & 0x1F) << 3
        b = ((word >> 11) & 0x1F) << 3
        opaque = bool(word & 0x0020)
        colours.append((r, g, b, opaque))
    return colours


def resize_with_center_crop(
    frame_bgr: np.ndarray, target_w: int, target_h: int
) -> np.ndarray:
    src_h, src_w = frame_bgr.shape[:2]
    scale = max(target_w / src_w, target_h / src_h)
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))

    resized = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    x = max(0, (scaled_w - target_w) // 2)
    y = max(0, (scaled_h - target_h) // 2)
    return resized[y : y + target_h, x : x + target_w]


# ---------------------------------------------------------------------------
# Fixed tile generation
# ---------------------------------------------------------------------------

def _encode_tile_packed(pixels: np.ndarray) -> bytes:
    """Encode an 8×8 index array to 32-byte VGA mode-2 packed format."""
    out = bytearray(32)
    for row in range(TILE_H):
        for col in range(TILE_W // 2):
            lo = int(pixels[row, col * 2]) & 0xF
            hi = int(pixels[row, col * 2 + 1]) & 0xF
            out[row * 4 + col] = lo | (hi << 4)
    return bytes(out)


def generate_tile_binaries(out_dir: str = "images") -> tuple[bytes, bytes]:
    """
    Build the two fixed tile dictionaries (256 tiles each, 8192 bytes each).

    Returns (tiles1_bytes, tiles2_bytes); also writes them to *out_dir*.
    """
    tiles1 = bytearray()
    tiles2 = bytearray()

    for N in range(256):
        H = (N >> 4) & 0xF
        L = N & 0xF

        p1 = np.zeros((TILE_H, TILE_W), dtype=np.uint8)
        p1[:4, 0::2] = H
        p1[4:, 0::2] = L

        p2 = np.zeros((TILE_H, TILE_W), dtype=np.uint8)
        p2[:4, 1::2] = H
        p2[4:, 1::2] = L

        tiles1 += _encode_tile_packed(p1)
        tiles2 += _encode_tile_packed(p2)

    assert len(tiles1) == 8192 and len(tiles2) == 8192

    os.makedirs(out_dir, exist_ok=True)
    for name, data in (("tiles1.bin", tiles1), ("tiles2.bin", tiles2)):
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"  Written {path} ({len(data)} bytes)")

    return bytes(tiles1), bytes(tiles2)


# ---------------------------------------------------------------------------
# Per-frame encoding
# ---------------------------------------------------------------------------

def encode_tile_map(
    base_img_idx: np.ndarray,
    ov_img_idx: np.ndarray,
    base_palette: np.ndarray,
    overlay_palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive 40×30 tile maps from per-pixel palette assignments.

    For each 8×8 tile block the "top" (H) and "bottom" (L) colour indices are
    found by averaging the active-column RGB values and snapping to the nearest
    palette entry.  The resulting tile ID = (H << 4) | L.

    Args:
        base_img_idx:    320×240 uint8, values 0-15.  Odd cols are forced to 0.
        ov_img_idx:      320×240 uint8, values 0-15.  Even cols are forced to 0;
                         0 means transparent for the overlay layer.
        base_palette:    (16, 3) uint8 RGB, Layer 1.
        overlay_palette: (16, 3) uint8 RGB, Layer 2 (index 0 = transparent black).
    """
    base_map = np.zeros((ROWS, COLS), dtype=np.uint8)
    overlay_map = np.zeros((ROWS, COLS), dtype=np.uint8)

    bp = base_palette.astype(np.int32)
    op = overlay_palette[1:].astype(np.int32)  # indices 1..15 only (skip transparent)

    for r in range(ROWS):
        for c in range(COLS):
            rs, cs = r * TILE_H, c * TILE_W

            # ---- Base layer (active = even screen cols within tile) ----
            block_b = base_img_idx[rs : rs + TILE_H, cs : cs + TILE_W]

            # Mean RGB of active pixels in top half and bottom half
            top_b_rgb = base_palette[block_b[:4, 0::2]].reshape(-1, 3).astype(np.float32).mean(0)
            bot_b_rgb = base_palette[block_b[4:, 0::2]].reshape(-1, 3).astype(np.float32).mean(0)

            H = int(np.argmin(np.sum((bp - top_b_rgb.astype(np.int32)) ** 2, axis=1)))
            L = int(np.argmin(np.sum((bp - bot_b_rgb.astype(np.int32)) ** 2, axis=1)))
            base_map[r, c] = (H << 4) | L

            # ---- Overlay layer (active = odd screen cols within tile) ----
            block_o = ov_img_idx[rs : rs + TILE_H, cs : cs + TILE_W]

            # odd-col values are 1..15; safe to index directly into overlay_palette
            top_o_rgb = overlay_palette[block_o[:4, 1::2]].reshape(-1, 3).astype(np.float32).mean(0)
            bot_o_rgb = overlay_palette[block_o[4:, 1::2]].reshape(-1, 3).astype(np.float32).mean(0)

            # Search only within indices 1..15 (+1 offset to restore true index)
            H_o = int(np.argmin(np.sum((op - top_o_rgb.astype(np.int32)) ** 2, axis=1))) + 1
            L_o = int(np.argmin(np.sum((op - bot_o_rgb.astype(np.int32)) ** 2, axis=1))) + 1
            overlay_map[r, c] = (H_o << 4) | L_o

    return base_map, overlay_map


def encode_frame(frame_bgr: np.ndarray, prev_centers=None) -> tuple[bytes, np.ndarray]:
    """
    Encode one frame to ML62 per-frame payload (2,464 bytes).

    Palette generation is identical to V3 (luma-split 31-colour).
    Tile maps are derived analytically using the fixed tile scheme.

    Returns (frame_bytes, raw_centers) where raw_centers can be passed as
    prev_centers on the next call to warm-start KMeans.
    """
    # 1. Spatial binning: 2×2 → 160×120 then back to 320×240 via nearest
    small = cv2.resize(frame_bgr, (SCREEN_W // 2, SCREEN_H // 2),
                       interpolation=cv2.INTER_AREA)
    binned = cv2.resize(small, (SCREEN_W, SCREEN_H),
                        interpolation=cv2.INTER_NEAREST)
    frame_rgb = cv2.cvtColor(binned, cv2.COLOR_BGR2RGB)

    # 2. 31-colour palette via KMeans on the binned small frame
    pixels = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
    init = prev_centers if (prev_centers is not None and len(prev_centers) == 31) else "k-means++"
    km = KMeans(n_clusters=31, n_init=1, init=init, random_state=42, max_iter=30)
    km.fit(pixels)
    raw_centers = km.cluster_centers_.copy()

    # Zero out empty clusters to suppress stray colour noise
    counts = np.bincount(km.labels_, minlength=31)
    for i in range(31):
        if counts[i] == 0:
            raw_centers[i] = [0.0, 0.0, 0.0]
    centers = raw_centers.clip(0, 255).astype(np.uint8)

    # Sort by luma then split into alternating even/odd brightness bands
    luma = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
    centers = centers[np.argsort(luma)]

    base_palette = centers[0::2]                      # 16 entries (even luma ranks)
    overlay_palette = np.zeros((16, 3), dtype=np.uint8)
    overlay_palette[1:] = centers[1::2]               # 15 entries; index 0 stays black/transparent

    # 3. Assign each pixel to its nearest palette entry
    #    Base layer: minimise over base_palette; force odd cols to 0
    diff_b = frame_rgb.astype(np.int32)[:, :, None, :] \
             - base_palette.astype(np.int32)[None, None, :, :]
    base_img_idx = np.argmin(np.sum(diff_b ** 2, axis=3), axis=2).astype(np.uint8)
    base_img_idx[:, 1::2] = 0

    #    Overlay layer: minimise over overlay_palette[1:]; offset by 1; force even cols to 0
    diff_o = frame_rgb.astype(np.int32)[:, :, None, :] \
             - overlay_palette[1:].astype(np.int32)[None, None, :, :]
    ov_img_idx = (np.argmin(np.sum(diff_o ** 2, axis=3), axis=2) + 1).astype(np.uint8)
    ov_img_idx[:, 0::2] = 0

    # 4. Derive tile maps analytically (no clustering needed — tiles are fixed)
    base_map, overlay_map = encode_tile_map(
        base_img_idx, ov_img_idx, base_palette, overlay_palette
    )

    # 5. Pack frame payload
    pal1_bytes = palette_to_bytes(base_palette)
    pal2_bytes = palette_to_bytes(overlay_palette, transparent_index0=True)
    map1_bytes = bytes(base_map.flatten())
    map2_bytes = bytes(overlay_map.flatten())

    return pal1_bytes + pal2_bytes + map1_bytes + map2_bytes, raw_centers


# ---------------------------------------------------------------------------
# Software reconstruction (verify / preview)
# ---------------------------------------------------------------------------

def reconstruct_frame_lb(frame_bytes: bytes) -> np.ndarray:
    """
    Decode an ML62 per-frame payload back to a BGR image.

    Tile pixels are derived from the mathematically fixed tile scheme —
    no tile binary files are needed.
    """
    pal1_data = frame_bytes[0:32]
    pal2_data = frame_bytes[32:64]
    map1_data = frame_bytes[64:1264]
    map2_data = frame_bytes[1264:2464]

    pal1 = parse_rgb555_palette(pal1_data)   # list of (r, g, b, opaque)
    pal2 = parse_rgb555_palette(pal2_data)

    map1 = np.frombuffer(map1_data, np.uint8).reshape(ROWS, COLS)
    map2 = np.frombuffer(map2_data, np.uint8).reshape(ROWS, COLS)

    img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    for r in range(ROWS):
        for c in range(COLS):
            N = int(map1[r, c])
            M = int(map2[r, c])

            H_base = N >> 4
            L_base = N & 0xF
            H_ov = M >> 4
            L_ov = M & 0xF

            for tr in range(TILE_H):
                row = r * TILE_H + tr
                b_idx = H_base if tr < 4 else L_base
                o_idx = H_ov if tr < 4 else L_ov

                b_color = pal1[b_idx][:3]
                o_opaque = bool(pal2[o_idx][3]) if o_idx < 16 else False
                o_color = pal2[o_idx][:3]

                for tc in range(TILE_W):
                    col = c * TILE_W + tc
                    if tc % 2 == 0:
                        # Even col: base layer (always opaque)
                        img[row, col] = b_color
                    else:
                        # Odd col: overlay if opaque, else base
                        img[row, col] = o_color if o_opaque else b_color

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_generate_tiles(args):
    print(f"Generating fixed tile dictionaries in '{args.out_dir}/'...")
    generate_tile_binaries(args.out_dir)
    print("Done.")


def cmd_encode(args):
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    start_frame = int((parse_time(args.start) if args.start else 0.0) * source_fps)
    end_frame = (
        int(parse_time(args.end) * source_fps)
        if args.end
        else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    )

    total_target_frames = int((end_frame - start_frame) / source_fps * args.fps)
    src_step = source_fps / args.fps

    print(f"Encoding {args.input}  →  {args.output}")
    print(f"Clip:    {args.start or '0'}  →  {args.end or 'end'}")
    print(f"Target:  {args.fps} FPS, ~{total_target_frames} frames")
    print(f"Format:  ML62  ({LB_FRAME_BYTES} bytes/frame)")
    print()

    with open(args.output, "wb") as fout:
        fout.write(
            struct.pack(
                "<4sBBHHHHI",
                HEADER_MAGIC, HEADER_VERSION, args.fps,
                SCREEN_W, SCREEN_H, TILE_W, TILE_H,
                total_target_frames,
            )
        )

        prev_centers = None
        t0 = time.time()
        for i in range(total_target_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + int(i * src_step))
            ret, frame = cap.read()
            if not ret:
                break

            frame_data, prev_centers = encode_frame(
                resize_with_center_crop(frame, SCREEN_W, SCREEN_H), prev_centers
            )
            fout.write(frame_data)

            elapsed = time.time() - t0
            fps_enc = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{(i + 1) / total_target_frames * 100:5.1f}%]  "
                f"frame {i + 1}/{total_target_frames}  "
                f"enc: {fps_enc:.1f} fps",
                end="\r",
            )

    cap.release()
    print(f"\n\nDone!  Saved {args.output}")

    if args.write_tiles:
        out_dir = os.path.dirname(args.output) or "."
        print(f"Writing tile dictionaries to '{out_dir}/'...")
        generate_tile_binaries(out_dir)


def cmd_verify(args):
    os.makedirs(args.debug_dir, exist_ok=True)

    with open(args.stream, "rb") as f:
        header = f.read(18)
        if len(header) < 18:
            print("ERROR: file too short for header", file=sys.stderr)
            sys.exit(1)

        magic, version, fps, w, h, tw, th, frame_count = struct.unpack(
            "<4sBBHHHHI", header
        )
        print(f"Magic:       {magic}")
        print(f"Version:     {version}")
        print(f"FPS:         {fps}")
        print(f"Resolution:  {w}×{h}")
        print(f"Tile size:   {tw}×{th}")
        print(f"Frames:      {frame_count}")
        print(f"Stream size: {LB_FRAME_BYTES * frame_count:,} bytes "
              f"({frame_count} × {LB_FRAME_BYTES})")
        print()

        if magic != HEADER_MAGIC:
            print(f"WARNING: expected magic {HEADER_MAGIC!r}, got {magic!r}")

        pil_frames = []
        for i in range(min(args.frames, frame_count)):
            data = f.read(LB_FRAME_BYTES)
            if len(data) < LB_FRAME_BYTES:
                print(f"Truncated at frame {i}")
                break

            recon = reconstruct_frame_lb(data)
            out_path = os.path.join(args.debug_dir, f"frame_{i:04d}.png")
            cv2.imwrite(out_path, recon)

            rgb = cv2.cvtColor(recon, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))
            print(f"  Saved {out_path}")

        if pil_frames:
            gif_path = os.path.join(args.debug_dir, "animation.gif")
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / fps) if fps > 0 else 41,
                loop=0,
            )
            print(f"\nSaved animated GIF to {gif_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MovieTime6502 LB (Low-Bandwidth) ML62 encoder"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- generate-tiles ----
    gen = sub.add_parser(
        "generate-tiles",
        help="Write TILES1.BIN and TILES2.BIN (fixed tile dictionaries)",
    )
    gen.add_argument(
        "--out-dir", default="images",
        help="Directory to write tiles1.bin / tiles2.bin (default: images/)"
    )

    # ---- encode ----
    enc = sub.add_parser("encode", help="Encode a video clip to ML62 format")
    enc.add_argument("--input",  default="Sprites/Metropolis_1927.mp4",
                     help="Source MP4 file")
    enc.add_argument("--start",  default=None,
                     help="Start timestamp HH:MM:SS or seconds (default: start of video)")
    enc.add_argument("--end",    default=None,
                     help="End timestamp HH:MM:SS or seconds (default: end of video)")
    enc.add_argument("--output", default="Movies/MOVIE_lb.BIN",
                     help="Output ML62 binary")
    enc.add_argument("--fps",    type=int, default=24, help="Target FPS")
    enc.add_argument("--write-tiles", action="store_true",
                     help="Also write TILES1.BIN / TILES2.BIN to the same directory as --output")

    # ---- verify ----
    ver = sub.add_parser("verify", help="Decode and inspect an ML62 stream")
    ver.add_argument("stream", help="ML62 binary stream file")
    ver.add_argument("--debug-dir", default="debug_verify_lb",
                     help="Directory for output PNGs / GIF")
    ver.add_argument("--frames",    type=int, default=10,
                     help="Number of frames to decode (default: 10)")

    args = parser.parse_args()
    if args.cmd == "generate-tiles":
        cmd_generate_tiles(args)
    elif args.cmd == "encode":
        cmd_encode(args)
    elif args.cmd == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
