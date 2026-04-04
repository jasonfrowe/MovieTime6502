#!/usr/bin/env python3
"""
generate_showcase_v6.py - Second Reality style 3D fly-through showcase for MovieTime6502.

Visual plan:
  - Layer 1 (MIDDLE / base): Scrolling 3D voxel terrain with a sky gradient.
  - Layer 2 (TOP / overlay): Rotating flat-shaded 3D octahedron in the sky.

Packing plan:
  - Renders 160x120 indexed images and upscales to 320x240 (2x2 pixel blocks).
  - The blocky nature drastically reduces the unique tile count, allowing
    lossless fast-path clustering into the 256-tile dictionary budget.

Usage:
    python tools/generate_showcase_v6.py --output SHOWCASE_V6.BIN
    python tools/generate_showcase_v6.py --output MOVIE.BIN --seconds 60 --fps 24
"""

import argparse
import struct
import time
from pathlib import Path
import os
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
        result[tile_id * 32:(tile_id + 1) * 32] = encoded
    return bytes(result)


def make_base_palette_v6() -> np.ndarray:
    """Classic DOS VGA landscape: Sky, Water, Grass, Mountains, Snow."""
    return np.array([
        (10, 15, 30),   # 0 Sky top
        (20, 30, 50),   # 1 Sky mid
        (30, 45, 75),   # 2 Sky low
        (45, 65, 100),  # 3 Sky horizon
        (15, 50, 120),  # 4 Water deep
        (25, 80, 160),  # 5 Water mid
        (35, 110, 200), # 6 Water shallow
        (20, 45, 20),   # 7 Grass dark
        (35, 70, 30),   # 8 Grass mid
        (55, 100, 45),  # 9 Grass light
        (80, 130, 60),  # 10 Grass bright
        (100, 90, 70),  # 11 Rock dark
        (130, 120, 100),# 12 Rock mid
        (170, 160, 140),# 13 Rock light
        (210, 210, 210),# 14 Snow
        (255, 255, 255) # 15 Snow bright
    ], dtype=np.uint8)


def make_overlay_palette_v6() -> np.ndarray:
    """Synthwave neon 3D object palette."""
    return np.array([
        (0, 0, 0),      # 0 Transparent
        (30, 0, 20),    # 1 Dark purple
        (50, 0, 35),
        (80, 0, 55),
        (120, 0, 80),   # 4 Magenta
        (160, 10, 100),
        (200, 30, 120),
        (240, 60, 140), # 7 Hot pink
        (255, 100, 120),
        (255, 140, 100),# 9 Orange
        (255, 180, 80),
        (255, 210, 60), # 11 Yellow
        (255, 230, 100),
        (255, 245, 150),
        (255, 255, 200),
        (255, 255, 255) # 15 White core
    ], dtype=np.uint8)


def find_tiles(indexed_img: np.ndarray, n_tiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster 1200 8x8 patches to n_tiles dictionary and tile map."""
    tiles = []
    for r in range(ROWS):
        for c in range(COLS):
            patch = indexed_img[r*TILE_H:(r+1)*TILE_H, c*TILE_W:(c+1)*TILE_W].astype(np.float32)
            tiles.append(patch.flatten())
    tiles_arr = np.array(tiles)

    unique_tiles, inverse_indices = np.unique(tiles_arr, axis=0, return_inverse=True)
    if len(unique_tiles) <= n_tiles:
        tile_dict = np.zeros((n_tiles, TILE_H, TILE_W), dtype=np.uint8)
        tile_dict[:len(unique_tiles)] = unique_tiles.reshape(-1, TILE_H, TILE_W).astype(np.uint8)
        return tile_dict, inverse_indices.reshape(ROWS, COLS).astype(np.uint8)

    km = KMeans(n_clusters=n_tiles, n_init=1, max_iter=30, random_state=42)
    tile_ids = km.fit_predict(tiles_arr)

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


def generate_base_highres_v6(frame_idx: int) -> np.ndarray:
    """Render sky and 3D voxel terrain into a 160x120 index image (0-15)."""
    t = frame_idx * 0.4
    cam_x = np.sin(t * 0.05) * 15.0
    cam_y = t * 1.5

    num_z = 70
    z = np.geomspace(1.0, 70.0, num_z)[:, None]  # Depth slices (70, 1)
    x_ang = np.linspace(-1.2, 1.2, 160)[None, :] # Screen columns (1, 160)

    wx = cam_x + x_ang * z
    wy = cam_y + z

    # Heightmap generation (mountains + noise)
    h = np.sin(wx * 0.2) * np.cos(wy * 0.2) * 8.0
    h += np.sin(wx * 0.05 + wy * 0.1) * 12.0
    h += np.sin(wx * 0.8) * np.cos(wy * 0.7) * 1.5

    # Flatten water level
    water_level = -2.0
    is_water = h < water_level
    h_render = h.copy()
    h_render[is_water] = water_level
    h_render[is_water] += np.sin(wx[is_water] * 2.0 + t * 1.5) * 0.4 # slight waves

    # Dynamically adjust camera height so we don't clip through terrain
    local_h = np.sin(cam_x * 0.2) * np.cos(cam_y * 0.2) * 8.0 + np.sin(cam_x * 0.05 + cam_y * 0.1) * 12.0
    cam_z = max(local_h + 8.0, water_level + 10.0) + np.sin(t * 0.1) * 2.0

    # Map heights to color palette indices
    c = np.zeros((num_z, 160), dtype=np.uint8)
    water_depth = water_level - h
    c[is_water] = np.clip(6 - (water_depth[is_water] * 0.8), 4, 6)
    c[~is_water] = np.clip(7 + (h_render[~is_water] - water_level) * 0.4, 7, 15)

    # Project 3D landscape to 2D screen Y coordinates
    horizon_y = 50.0 + np.sin(t * 0.08) * 8.0
    sy = ((cam_z - h_render) / z * 35.0 + horizon_y).astype(np.int32)

    img = np.zeros((120, 160), dtype=np.uint8)
    Y_screen = np.arange(120)[:, None]

    # Render base sky gradient
    sky_t = (Y_screen - horizon_y + 40) / 10.0
    sky_bands = np.clip(np.floor(sky_t), 0, 3).astype(np.uint8)
    img[:] = sky_bands

    # Painter's Algorithm: Draw from back to front using vectorized broadcasting
    c_broadcast = np.broadcast_to(c, (num_z, 160))
    for i in range(num_z - 1, -1, -1):
        mask = Y_screen >= sy[i]
        img[mask] = np.broadcast_to(c_broadcast[i], (120, 160))[mask]

    return img


def generate_overlay_highres_v6(frame_idx: int) -> np.ndarray:
    """Render a rotating 3D flat-shaded octahedron into a 160x120 overlay image."""
    img = np.zeros((120, 160), dtype=np.uint8)
    t = frame_idx * 0.05

    verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=np.float32)

    faces = [
        (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
        (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5)
    ]

    # Apply 3D Rotation
    rx, ry, rz = t * 0.7, t * 1.1, t * 0.5
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    scale = 15.0 + np.sin(t * 1.5) * 3.0
    v_rot = verts @ (Rz @ Ry @ Rx).T * scale

    # Translate Object in 3D Space
    v_rot[:, 0] += np.sin(t * 0.5) * 30.0         # X Hover
    v_rot[:, 1] += -15.0 + np.cos(t * 0.8) * 12.0 # Y Bobbing
    v_rot[:, 2] += 60.0 + np.sin(t * 0.3) * 20.0  # Z Depth

    light_dir = np.array([0.5, -0.6, -0.6])
    light_dir /= np.linalg.norm(light_dir)

    fov = 90.0
    for face in faces:
        p0, p1, p2 = v_rot[face[0]], v_rot[face[1]], v_rot[face[2]]

        # Backface Culling
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len == 0: continue
        normal /= norm_len

        view_dir = p0 / np.linalg.norm(p0)
        if np.dot(normal, view_dir) > 0.05:
            continue

        # Flat Shading
        intensity = np.clip(np.dot(normal, -light_dir), 0.0, 1.0)
        color = int(1 + intensity * 14)

        # Projection
        pts = []
        for p in [p0, p1, p2]:
            z = max(p[2], 1.0)
            pts.append([int(p[0] * fov / z + 80), int(p[1] * fov / z + 60)])

        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], isClosed=True, color=15, thickness=1) # wire highlight

    return img


def build_frame(frame_idx: int, fps: int) -> bytes:
    _ = fps
    
    overlay_img_binned = generate_overlay_highres_v6(frame_idx)
    base_img_binned = generate_base_highres_v6(frame_idx)
    
    # Expand 160x120 native resolutions to standard 320x240 for 8x8 tile processing
    base_img = np.repeat(np.repeat(base_img_binned, 2, axis=0), 2, axis=1)
    overlay_img = np.repeat(np.repeat(overlay_img_binned, 2, axis=0), 2, axis=1)

    base_tiles, base_map = find_tiles(base_img, NUM_TILES)
    over_tiles, over_map = find_tiles(overlay_img, NUM_TILES)

    return (
        palette_to_bytes(make_base_palette_v6())
        + palette_to_bytes(make_overlay_palette_v6(), transparent_index0=True)
        + build_tileset(over_tiles)
        + build_tileset(base_tiles)
        + bytes(over_map.flatten())
        + bytes(base_map.flatten())
    )


def generate_stream(output_path: Path, seconds: float, fps: int) -> None:
    frame_count = int(round(seconds * fps))
    if frame_count <= 0:
        raise ValueError("seconds * fps must produce at least one frame")

    print("Generating V6 Second Reality 3D Fly-through showcase")
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
    
    # Layer 1 (Base/Middle)
    for r in range(ROWS):
        for c in range(COLS):
            tile = tiles1[map1[r, c]]
            for tr in range(TILE_H):
                for tc in range(TILE_W):
                    idx = tile[tr, tc]
                    img[r*TILE_H+tr, c*TILE_W+tc] = pal1[idx][:3]

    # Layer 2 (Overlay/Top)
    for r in range(ROWS):
        for c in range(COLS):
            tile = tiles2[map2[r, c]]
            for tr in range(TILE_H):
                for tc in range(TILE_W):
                    idx = tile[tr, tc]
                    if pal2[idx][3]:
                        img[r*TILE_H+tr, c*TILE_W+tc] = pal2[idx][:3]

    return img


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
            cv2.imwrite(out_path, cv2.cvtColor(recon, cv2.COLOR_RGB2BGR))
            
            pil_frames.append(Image.fromarray(recon))
            print(f"Saved debug frame {i}")
            
        if pil_frames:
            gif_path = os.path.join(args.debug_dir, "animation.gif")
            pil_frames[0].save(
                gif_path, save_all=True, append_images=pil_frames[1:],
                duration=int(1000 / fps) if fps > 0 else 41, loop=0
            )
            print(f"\nSaved animated GIF to {gif_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V6 Second Reality Fly-through showcase generator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="Generate the showcase into MT62 format")
    enc.add_argument("--output", default="SHOWCASE_V6.BIN", help="Output stream filename")
    enc.add_argument("--seconds", type=float, default=60.0, help="Length of the showcase in seconds")
    enc.add_argument("--fps", type=int, default=24, help="Playback FPS (default: 24)")

    ver = sub.add_parser("verify", help="Decode and inspect a packed stream")
    ver.add_argument("stream", help="MT62 binary stream file")
    ver.add_argument("--debug-dir", default="debug_verify_v6", help="Output dir for PNGs")
    ver.add_argument("--frames", type=int, default=10, help="Number of frames to decode")

    args = parser.parse_args()
    if args.cmd == "encode":
        generate_stream(Path(args.output), args.seconds, args.fps)
    elif args.cmd == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()