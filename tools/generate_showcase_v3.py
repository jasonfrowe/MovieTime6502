#!/usr/bin/env python3
"""
generate_showcase_v3.py — "Plasma Storm" MT62 showcase for MovieTime6502.

V3 improvements over V2
-----------------------
1. Base tiles: all 256 slots are genuinely distinct.
   - 16 brightness levels × 16 clearly different 8×8 textures (solid, three
     periods of H/V/diagonal stripes, 2×2 checker, 2×2 dots, ring, diamond,
     plus, X, outer frame, brick).  No budget wasted on near-duplicate
     rotated sine waves.

2. Base map: two fully decorrelated multi-frequency plasmas drive the high
   nibble (brightness 0-15) and the low nibble (pattern 0-15) independently,
   so the entire 256-tile vocabulary is accessed uniformly across the screen.

3. Overlay: instead of sparse, flickering cyber-grid symbols, the overlay
   uses a stable "shape grid" that is revealed by expanding sonar-ring pulses
   from 3 sinusoidally-animated Lissajous sources.  The shape at each tile
   cell is fixed for that position (no per-frame shape change = no flicker).
   Only brightness changes with ring proximity → zero tile churn, all 256
   overlay tile IDs in use.

4. Tile data is pre-built ONCE (constant content across frames), so the
   per-frame cost is just two numpy plasma evaluations + ring maths +
   two palette generations.  Generation is significantly faster than V2.

Usage:
    python tools/generate_showcase_v3.py --output SHOWCASE_V3.BIN
    python tools/generate_showcase_v3.py --output MOVIE.BIN --seconds 60 --fps 24
"""

import argparse
import colorsys
import struct
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (must match player / MT62 format)
# ---------------------------------------------------------------------------
SCREEN_W = 320
SCREEN_H = 240
TILE_W   = 8
TILE_H   = 8
COLS = SCREEN_W // TILE_W   # 40
ROWS = SCREEN_H // TILE_H   # 30
NUM_TILES = 256
COLOURS   = 16               # palette entries per layer (4-bit index)

PALETTE_BYTES = COLOURS * 2
TILES_BYTES   = NUM_TILES * TILE_H * (TILE_W // 2)   # 8 192
MAP_BYTES     = COLS * ROWS                           # 1 200
FRAME_BYTES   = (PALETTE_BYTES * 2) + (TILES_BYTES * 2) + (MAP_BYTES * 2)

HEADER_MAGIC   = b"MT62"
HEADER_VERSION = 1

# ---------------------------------------------------------------------------
# Sonar ring parameters
# ---------------------------------------------------------------------------
NUM_SOURCES      = 3    # independent pulse sources
RINGS_PER_SOURCE = 3    # concurrent ring ages per source
PULSE_PERIOD     = 72   # frames per full ring cycle (per source)
PULSE_SPEED      = 0.35  # tile-widths expanded per frame
PULSE_WIDTH      = 0.9  # half-width of ring in tile-widths
RING_MAX_RADIUS  = 27.0  # rings fade out at this tile radius
RING_MIN_BRIGHT  = 0.20  # brightness below which a cell stays transparent

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def rgb8_to_rgb555(r: int, g: int, b: int, opaque: bool = True) -> int:
    word = ((b >> 3) << 11) | ((g >> 3) << 6) | (r >> 3)
    if opaque:
        word |= 0x0020
    return word


def palette_to_bytes(palette_rgb: np.ndarray, transparent_index0: bool = False) -> bytes:
    out = bytearray()
    for idx, (r, g, b) in enumerate(palette_rgb):
        word = rgb8_to_rgb555(
            int(r), int(g), int(b),
            opaque=not (transparent_index0 and idx == 0),
        )
        out += struct.pack("<H", word)
    return bytes(out)


def encode_tile(tile_8x8_idx: np.ndarray) -> bytes:
    out = bytearray(32)
    for row in range(TILE_H):
        for col in range(TILE_W // 2):
            lo = int(tile_8x8_idx[row, col * 2])     & 0xF
            hi = int(tile_8x8_idx[row, col * 2 + 1]) & 0xF
            out[row * 4 + col] = lo | (hi << 4)
    return bytes(out)


def build_tileset_bytes(tiles: np.ndarray) -> bytes:
    """Encode (256, 8, 8) uint8 tile array into 8 192 raw bytes."""
    result = bytearray(TILES_BYTES)
    for i in range(NUM_TILES):
        result[i * 32:(i + 1) * 32] = encode_tile(tiles[i])
    return bytes(result)


# ---------------------------------------------------------------------------
# Base layer tile vocabulary — 16 brightness × 16 distinct patterns
# ---------------------------------------------------------------------------
# tile_id = (brightness << 4) | pattern_type
# brightness 0-15 → primary palette colour index
# pattern_type  0-15 → internal 8×8 texture (every pattern is visually distinct)
# Secondary colour is 5 steps below primary (noticeably darker) so stripes /
# patterns have strong internal contrast even at lower brightness levels.
# ---------------------------------------------------------------------------

def _make_base_tile(brightness: int, pattern: int) -> np.ndarray:
    c_hi = brightness
    c_lo = max(0, brightness - 5)

    # 2D meshgrid – broadcasts correctly to (TILE_H, TILE_W) = (8, 8)
    X, Y = np.meshgrid(
        np.arange(TILE_W, dtype=np.int32),
        np.arange(TILE_H, dtype=np.int32),
    )

    # Start fully filled with c_hi; selected cells become c_lo.
    tile = np.full((TILE_H, TILE_W), c_hi, dtype=np.uint8)

    if pattern == 0:   # solid — every pixel is c_hi
        pass
    elif pattern == 1: # horizontal stripes, 1 px period
        tile[Y % 2 == 0] = c_lo
    elif pattern == 2: # horizontal stripes, 2 px period
        tile[(Y // 2) % 2 == 0] = c_lo
    elif pattern == 3: # horizontal stripes, 4 px period
        tile[(Y // 4) % 2 == 0] = c_lo
    elif pattern == 4: # vertical stripes, 1 px period
        tile[X % 2 == 0] = c_lo
    elif pattern == 5: # vertical stripes, 2 px period
        tile[(X // 2) % 2 == 0] = c_lo
    elif pattern == 6: # vertical stripes, 4 px period
        tile[(X // 4) % 2 == 0] = c_lo
    elif pattern == 7: # 2×2 checkerboard
        tile[((X // 2) + (Y // 2)) % 2 == 0] = c_lo
    elif pattern == 8: # diagonal stripes  /  (slope +1)
        tile[(X + Y) % 4 < 2] = c_lo
    elif pattern == 9: # diagonal stripes  \  (slope -1)
        tile[(X - Y) % 4 < 2] = c_lo
    elif pattern == 10:# 2×2 dot grid (bright dots on darker field)
        tile[:] = c_lo
        tile[(X % 4 < 2) & (Y % 4 < 2)] = c_hi
    elif pattern == 11:# 1 px border ring
        tile[(X == 0) | (X == 7) | (Y == 0) | (Y == 7)] = c_lo
    elif pattern == 12:# diamond shape  (L1 distance mask)
        tile[np.abs(X - 3) + np.abs(Y - 3) > 3] = c_lo
    elif pattern == 13:# X (both diagonals highlighted)
        on_x = (X == Y) | (X == 7 - Y)
        tile[~on_x] = c_lo
    elif pattern == 14:# + cross
        on_plus = (X == 3) | (X == 4) | (Y == 3) | (Y == 4)
        tile[~on_plus] = c_lo
    elif pattern == 15:# brick (H-stripes + staggered vertical mortar line)
        # Even rows: mortar at x=0,4; odd rows: mortar at x=2,6
        is_odd_row = (Y % 2 == 1)
        even_mortar = (~is_odd_row) & (X % 4 == 0)
        odd_mortar  = is_odd_row  & ((X + 2) % 4 == 0)
        tile[is_odd_row | even_mortar | odd_mortar] = c_lo

    return tile


def build_base_tileset_v3() -> np.ndarray:
    """Return (256, 8, 8) uint8 — all 256 base tiles, built once."""
    tiles = np.empty((NUM_TILES, TILE_H, TILE_W), dtype=np.uint8)
    for tid in range(NUM_TILES):
        tiles[tid] = _make_base_tile((tid >> 4) & 0xF, tid & 0xF)
    return tiles


# ---------------------------------------------------------------------------
# Base layer map — two decorrelated multi-frequency plasmas
# ---------------------------------------------------------------------------

def generate_base_map_v3(frame_idx: int) -> np.ndarray:
    """
    Drives tile_id = (brightness_nibble << 4) | pattern_nibble.
    Using different, incommensurable spatial frequency sets for each nibble
    ensures the full 256-tile vocabulary is uniformly distributed across the
    screen (no clustering in a subset of tile IDs).
    """
    t  = frame_idx * 0.04
    cx = (np.arange(COLS, dtype=np.float32) + 0.5) / COLS   # 0..1
    cy = (np.arange(ROWS, dtype=np.float32) + 0.5) / ROWS
    X, Y = np.meshgrid(cx, cy)
    R    = np.hypot(X - 0.5, Y - 0.5)

    # Brightness plasma — 4 incommensurable frequencies
    a = np.sin(X * 11.0 + t *  1.30)
    b = np.sin(Y *  8.0 + t * -1.10)
    c = np.sin((X + Y) * 7.0 + t * 0.90)
    d = np.sin(R * 15.0 - t * 2.00)
    brightness_raw = (a + b + c + d) * 0.25   # -1..1

    # Pattern plasma — completely different frequencies (fully decorrelated)
    e = np.cos(X * 17.0 + t * -0.70)
    f = np.cos(Y * 13.0 + t *  1.30)
    g = np.cos((X - Y) * 11.0 + t * 0.50)
    pattern_raw = (e + f + g) / 3.0            # -1..1

    brightness = np.clip(np.floor((brightness_raw + 1.0) * 8.0), 0, 15).astype(np.uint8)
    pattern    = np.clip(np.floor((pattern_raw   + 1.0) * 8.0), 0, 15).astype(np.uint8)

    return ((brightness << 4) | pattern).astype(np.uint8)


# ---------------------------------------------------------------------------
# Overlay tile vocabulary — shape × brightness encoding
# ---------------------------------------------------------------------------
# tile_id = 0                → transparent (all pixels = 0)
# tile_id = (shape << 4) | b → shape 0-15, brightness b 1-15
#           brightness 0    → transparent tile (no pixels set)
#           brightness 1-15 → pixels active at overlay palette index b
#
# 16 shapes × 15 brightness steps = 240 non-transparent tiles
# plus tile 0 (transparent) = 241 occupied out of 256 slots.
# ---------------------------------------------------------------------------

def _make_overlay_tile(shape: int, brightness: int) -> np.ndarray:
    if brightness == 0:
        return np.zeros((TILE_H, TILE_W), dtype=np.uint8)

    c  = min(15, max(1, brightness))   # never use index 0 (transparent)
    X, Y = np.meshgrid(
        np.arange(TILE_W, dtype=np.int32),
        np.arange(TILE_H, dtype=np.int32),
    )
    tile = np.zeros((TILE_H, TILE_W), dtype=np.uint8)

    # Plasma-like masks that mirror the base tile vocabulary so the ring reads
    # as a highlight of the existing texture rather than a separate HUD layer.
    if shape == 0:
        mask = ((X + Y) % 2) == 0
    elif shape == 1:
        mask = (Y % 2) == 1
    elif shape == 2:
        mask = ((Y // 2) % 2) == 1
    elif shape == 3:
        mask = ((Y // 4) % 2) == 1
    elif shape == 4:
        mask = (X % 2) == 1
    elif shape == 5:
        mask = ((X // 2) % 2) == 1
    elif shape == 6:
        mask = ((X // 4) % 2) == 1
    elif shape == 7:
        mask = (((X // 2) + (Y // 2)) % 2) == 1
    elif shape == 8:
        mask = ((X + Y) % 4) >= 2
    elif shape == 9:
        mask = ((X - Y) % 4) >= 2
    elif shape == 10:
        mask = (X % 4 < 2) & (Y % 4 < 2)
    elif shape == 11:
        mask = (X == 0) | (X == 7) | (Y == 0) | (Y == 7)
    elif shape == 12:
        mask = (np.abs(X - 3) + np.abs(Y - 3)) <= 3
    elif shape == 13:
        mask = (X == Y) | (X == 7 - Y)
    elif shape == 14:
        mask = (X == 3) | (X == 4) | (Y == 3) | (Y == 4)
    else:
        is_odd_row = (Y % 2) == 1
        even_mortar = (~is_odd_row) & ((X % 4) == 0)
        odd_mortar = is_odd_row & (((X + 2) % 4) == 0)
        mask = ~(is_odd_row | even_mortar | odd_mortar)

    tile[mask] = c
    return tile.astype(np.uint8)


def build_overlay_tileset_v3() -> np.ndarray:
    """Return (256, 8, 8) uint8 — constant overlay tile vocabulary."""
    tiles = np.zeros((NUM_TILES, TILE_H, TILE_W), dtype=np.uint8)
    # tile 0 remains all-zeros (transparent)
    for tid in range(1, NUM_TILES):
        shape      = (tid >> 4) & 0xF
        brightness = tid & 0xF
        tiles[tid] = _make_overlay_tile(shape, brightness)
    return tiles


# ---------------------------------------------------------------------------
# Overlay layer map — sonar rings from 3 Lissajous-animated sources
# ---------------------------------------------------------------------------
# A fixed "shape grid" is keyed by tile position (stable = no flicker).
# Ring passes illuminate cells, changing their brightness index.
# Transparent (tile_id=0) everywhere not reached by a ring.
# ---------------------------------------------------------------------------

def generate_overlay_map_v3(frame_idx: int, base_pattern: np.ndarray) -> np.ndarray:
    """
    Returns (30, 40) uint8 tile-ID map.
    tile_id 0 → transparent; tile_id > 0 → (pattern<<4)|brightness.
    pattern nibble comes from the current base plasma map, so ring highlights
    inherit the same local texture orientation as the background.
    """
    CX = np.arange(COLS, dtype=np.float32)[None, :]   # (1, 40)
    CY = np.arange(ROWS, dtype=np.float32)[:, None]   # (30, 1)

    best_bright = np.zeros((ROWS, COLS), dtype=np.float32)

    cycle_len = max(1, PULSE_PERIOD // RINGS_PER_SOURCE)

    for s in range(NUM_SOURCES):
        # Each source follows a slow Lissajous path (120° phase apart)
        base_phase = s * 2.094   # 2π/3 between sources
        t_src = frame_idx * 0.022 + base_phase
        src_x = (COLS * 0.5) + (COLS * 0.36) * np.sin(t_src * (0.80 + s * 0.20))
        src_y = (ROWS * 0.5) + (ROWS * 0.36) * np.cos(t_src * (0.65 + s * 0.15))

        dist = np.hypot(CX - src_x, CY - src_y)   # (ROWS, COLS)

        for k in range(RINGS_PER_SOURCE):
            # Stagger rings evenly within the period
            ring_age = (frame_idx % cycle_len + k * cycle_len) % PULSE_PERIOD
            ring_radius = ring_age * PULSE_SPEED
            fade = max(0.0, 1.0 - ring_radius / RING_MAX_RADIUS)
            if fade <= 0.0:
                continue

            proximity = 1.0 - np.abs(dist - ring_radius) / PULSE_WIDTH
            proximity = np.clip(proximity, 0.0, 1.0) * fade

            best_bright = np.maximum(best_bright, proximity)

    # Map 0..1 float → brightness index 1-15; cells below threshold → 0 (transparent).
    # Remap [RING_MIN_BRIGHT, 1.0] → [1, 15] so even the ring edge is clearly visible.
    visible = best_bright >= RING_MIN_BRIGHT
    remapped = (best_bright - RING_MIN_BRIGHT) / max(1e-6, 1.0 - RING_MIN_BRIGHT)
    bright_idx = np.where(
        visible,
        np.clip(np.floor(remapped * 14.0) + 1, 1, 15),
        0,
    ).astype(np.uint8)

    # Combine current base pattern with animated brightness into tile_id
    tile_map = np.where(
        bright_idx > 0,
        ((base_pattern & 0xF) << 4) | bright_idx,
        np.uint8(0),
    ).astype(np.uint8)

    return tile_map


# ---------------------------------------------------------------------------
# Palettes — independent, fast-cycling colour schemes per layer
# ---------------------------------------------------------------------------

def make_base_palette_v3(frame_idx: int, fps: int) -> np.ndarray:
    """
    16-entry base palette: warm amber/orange/red tones.
    Stays in hue range 0.0-0.15 (red→orange→yellow) so it is always
    visually distinct from the cool-hue overlay.
    """
    phase = (frame_idx / max(fps, 1)) * 0.08  # slow drift
    palette = np.zeros((COLOURS, 3), dtype=np.uint8)
    for i in range(COLOURS):
        t   = i / COLOURS
        # Hue sweep: red (0.00) through orange (0.08) to yellow (0.15)
        hue = (t * 0.15 + phase * 0.10) % 1.0
        sat = 0.90 - 0.20 * t
        val = 0.20 + 0.80 * ((i + 1) / COLOURS)
        r, g, b = colorsys.hsv_to_rgb(
            hue, float(np.clip(sat, 0.0, 1.0)), float(np.clip(val, 0.0, 1.0))
        )
        palette[i] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def make_overlay_palette_v3(frame_idx: int, fps: int) -> np.ndarray:
    """
    16-entry overlay palette: warm highlights near the base palette hues.
    This intentionally blends with the plasma, so rings feel integrated rather
    than looking like a separate UI layer.
    Index 0 is placeholder (marked transparent by palette_to_bytes).
    """
    phase = (frame_idx / max(fps, 1)) * 0.10
    palette = np.zeros((COLOURS, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)   # placeholder for transparent entry
    for i in range(1, COLOURS):
        t   = (i - 1) / (COLOURS - 1)
        # Keep hues close to base (orange→amber), but a touch brighter.
        hue = (0.03 + t * 0.12 + phase * 0.04) % 1.0
        sat = 0.70 - 0.20 * t
        val = 0.55 + 0.40 * (0.5 + 0.5 * np.sin(phase * 4.0 + t * np.pi))
        r, g, b = colorsys.hsv_to_rgb(
            hue, float(np.clip(sat, 0.0, 1.0)), float(np.clip(val, 0.0, 1.0))
        )
        palette[i] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


# ---------------------------------------------------------------------------
# Frame assembly (tiles pre-built once outside the loop)
# ---------------------------------------------------------------------------

def build_frame(
    frame_idx: int,
    fps: int,
    base_tile_bytes: bytes,
    overlay_tile_bytes: bytes,
) -> bytes:
    pal1 = make_overlay_palette_v3(frame_idx, fps)   # overlay
    pal2 = make_base_palette_v3(frame_idx, fps)       # base

    map_base = generate_base_map_v3(frame_idx)  # (30, 40) uint8
    map_overlay = generate_overlay_map_v3(frame_idx, map_base & 0x0F)

    return (
        palette_to_bytes(pal2)                              # base palette    → Layer 1 slot (32 B)
        + palette_to_bytes(pal1, transparent_index0=True)  # overlay palette → Layer 2 slot (32 B)
        + overlay_tile_bytes                               # overlay tiles   → Layer 2 slot (8192 B)
        + base_tile_bytes                                  # base tiles      → Layer 1 slot (8192 B)
        + bytes(map_overlay.flatten())                     # overlay map     → Layer 2 slot (1200 B)
        + bytes(map_base.flatten())                        # base map        → Layer 1 slot (1200 B)
    )


# ---------------------------------------------------------------------------
# Stream generation
# ---------------------------------------------------------------------------

def generate_stream(output_path: Path, seconds: float, fps: int) -> None:
    frame_count = int(round(seconds * fps))
    if frame_count <= 0:
        raise ValueError("seconds × fps must be at least 1")

    print("Generating V3 Procedural MT62 Showcase — Plasma Storm")
    print(f"  Frames:  {frame_count}")
    print(f"  FPS:     {fps}")
    print(f"  Seconds: {seconds:.1f}")
    print(f"  Output:  {output_path}")
    print(f"  Payload: {FRAME_BYTES:,} bytes/frame")
    print()

    print("  Building tile dictionaries (once)…", end=" ", flush=True)
    t_tiles = time.time()
    base_tile_bytes    = build_tileset_bytes(build_base_tileset_v3())
    overlay_tile_bytes = build_tileset_bytes(build_overlay_tileset_v3())
    print(f"done in {time.time() - t_tiles:.2f}s")

    header = struct.pack(
        "<4sBBHHHHI",
        HEADER_MAGIC, HEADER_VERSION, fps,
        SCREEN_W, SCREEN_H, TILE_W, TILE_H,
        frame_count,
    )

    t0 = time.time()
    with output_path.open("wb") as fout:
        fout.write(header)
        for frame_idx in range(frame_count):
            fout.write(build_frame(frame_idx, fps, base_tile_bytes, overlay_tile_bytes))
            if frame_idx % fps == 0 or frame_idx == frame_count - 1:
                elapsed = time.time() - t0 + 1e-9
                rate    = (frame_idx + 1) / elapsed
                pct     = (frame_idx + 1) * 100.0 / frame_count
                print(
                    f"  [{pct:5.1f}%] frame {frame_idx + 1}/{frame_count}"
                    f"  {rate:.1f} fr/s",
                    end="\r",
                )

    elapsed = time.time() - t0
    size_mib = output_path.stat().st_size / 1024 / 1024
    print(f"\n  Done in {elapsed:.1f}s  →  {output_path}  ({size_mib:.1f} MiB)")
    print("\nCopy or rename to MOVIE.BIN to play on the RP6502 demo ROM.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate V3 procedural MT62 showcase stream (Plasma Storm)"
    )
    parser.add_argument("--output",  default="SHOWCASE_V3.BIN",
                        help="Output file (default: SHOWCASE_V3.BIN)")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="Duration in seconds (default: 60)")
    parser.add_argument("--fps",     type=int,   default=24,
                        help="Playback FPS (default: 24)")
    args = parser.parse_args()

    generate_stream(Path(args.output), args.seconds, args.fps)


if __name__ == "__main__":
    main()
