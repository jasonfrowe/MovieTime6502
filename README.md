# MovieTime6502

MovieTime6502 streams a pre-encoded tile movie from USB mass storage and plays
it on the RP6502 using VGA mode-2 dual tile layers at up to 24 fps.

## Project Layout

```
src/
  main.c          Runtime player — reads a .BIN file from USB, double-buffers
                  each 18,848-byte frame into XRAM, and presents it vsync-locked.
  constants.h     XRAM address map, frame format sizes, and header constants.

tools/
  encode_movie.py     Full-featured offline encoder with film-grain denoise,
                      temporal blend, and luma-based layer partition.
  encode_movie_v2.py  Simplified encoder — cleaner code, same luma partition fix.
  encode_movie_v3.py  Striped dither encoder — strictly partitions hardware layers
                      spatially using alternating columns for rock-solid mixing.
  encode_movie_v4.py  Binned encoder — 2x pixel binning (320×240 → 160×120)
                      before quantization; fastest encoder, most stable tiles.
  generate_showcase_v5.py  Procedural Amiga-style bouncing ball showcase — no
                            source video required.
```

---

## Build

Initialize the vendored SDK submodule once:

```
git submodule update --init --recursive
```

Build and install the SDK locally (one-time):

```
cmake -S external/llvm-mos-sdk -B external/llvm-mos-sdk/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/external/llvm-mos
cmake --build external/llvm-mos-sdk/build --target install -j4
```

Build the player ROM:

```
cmake -S . -B build/target-native
cmake --build build/target-native --target MovieTime6502
```

Output: `build/target-native/MovieTime6502.rp6502`

---

## Python Dependencies

```
pip install opencv-python numpy scikit-learn Pillow
```

---

## Encoders

All encoders produce the same MT62 format and use the same `encode` / `verify`
subcommands. Pick the one that fits your workflow.

### encode_movie.py — full-featured encoder

Best overall quality. Includes luma denoise, temporal blend, and per-pixel
overlay selection.

```
python tools/encode_movie.py encode \
  --input  Sprites/Metropolis_1927.mp4 \
  --start  "01:23:00" \
  --end    "01:26:00" \
  --output MOVIE.BIN \
  [--fps 24] \
  [--debug-dir debug_frames] \
  [--debug-every 24] \
  [--denoise-h 9] \
  [--sharpen 0.10] \
  [--temporal 0.50] \
  [--motion-thresh 8.0] \
  [--bright-threshold 0.0] \
  [--dark-lock-margin 10.0]
```

Key options:

| Flag | Default | Purpose |
|------|---------|---------|
| `--denoise-h` | 9 | Luma denoise strength (0 = off) |
| `--sharpen` | 0.10 | Unsharp mask after denoise |
| `--temporal` | 0.50 | Static-region temporal blend (reduces flicker) |
| `--motion-thresh` | 8.0 | Pixel delta below which a region is considered static |
| `--bright-threshold` | 0 | Drop pixels below this luma; 0 = off |
| `--dark-lock-margin` | 10 | Extra luma range above threshold forced to stable black |

### encode_movie_v2.py — simplified encoder

Identical MT62 output format. Cleaner, easier to read. No denoise pipeline.

```
python tools/encode_movie_v2.py \
  --input  Sprites/Metropolis_1927.mp4 \
  --start  "01:23:00" \
  --end    "01:26:00" \
  --output MOVIE.BIN \
  [--fps 24]
```

### encode_movie_v3.py — striped dither encoder 

Strictly partitions the hardware layers spatially to eliminate residual noise. The image is binned by 2×2, then a 31-color global palette is generated and split by brightness. The base layer draws ONLY the even columns, and the overlay layer draws ONLY the odd columns. This creates a flawless, rock-solid hardware dither. The resulting tile maps are more stable than the full encoder, but not as much as the binned V4 encoder.

```
python tools/encode_movie_v3.py encode \
--input Sprites/Metropolis_1927.mp4 \
--start "01:23:00" \
--end "01:26:00" \
--output MOVIE.BIN \
[--fps 24]
```

### encode_movie_v4.py — binned encoder

Bins the source 320×240 frame to 160×120 before quantization. Each screen
pixel becomes a solid 2×2 block, so each 8×8 tile is only a 4×4 grid of
distinct values. This drastically reduces unique tile patterns, speeds up
encoding (~8x faster than the full encoder), and produces the most temporally
stable tile maps.

```
python tools/encode_movie_v4.py encode \
  --input  Sprites/Metropolis_1927.mp4 \
  --start  "01:23:00" \
  --end    "01:26:00" \
  --output MOVIE.BIN \
  [--fps 24]
```

---

## Verify an Encoded Stream

Decode the first N frames to PNG for quality inspection without needing hardware:

```
python tools/encode_movie.py verify MOVIE.BIN --debug-dir debug_verify --frames 10
# or
python tools/encode_movie_v4.py verify MOVIE.BIN --debug-dir debug_verify --frames 10
```

---

## Procedural Showcase (no source video required)

Generates an Amiga-style bouncing checker ball over a perspective floor.
Layer 1 (base) draws the sky/floor/shadow; Layer 2 (overlay) draws the ball
with transparent index 0 so the background shows through.

```
python tools/generate_showcase_v5.py \
  --output SHOWCASE_V5.BIN \
  [--seconds 60] \
  [--fps 24]
```

---

## Run on Hardware

1. Copy the `.BIN` file to the root of your USB mass storage device.
2. The player defaults to `SHOWCASE.BIN`; pass a filename as an argument to
   play a different file (e.g. `MOVIE.BIN`).
3. Upload and run `MovieTime6502.rp6502`.
4. The player prints FPS and KB/s throughput stats every two seconds.

### Controls

| Action | Keyboard | Gamepad |
|--------|----------|---------|
| Play / Pause | `Space` or `Enter` | START |
| Stop (exit) | `Esc` or `Q` | SELECT |
| Fast Forward | `→` or `L` | D-Pad Right / R1 |
| Rewind | `←` or `J` | D-Pad Left / L1 |

Fast Forward and Rewind jump ~1 second (24 frames) per poll cycle while held.
Both are disabled while paused.

Gamepad button assignments can be remapped without recompiling by placing a
`JOYSTICK_CA.DAT` (preferred) or `JOYSTICK.DAT` file in the root of the USB
drive. The file format is identical to the one used by RPMegaRaider:

```
1 byte    num_mappings
N × 3 bytes  { uint8_t action_id, uint8_t field, uint8_t mask }
```

`action_id` values: `0` = Play/Pause, `1` = Stop, `2` = Fast Forward, `3` = Rewind.  
`field` values: `0` = D-Pad, `1` = Sticks, `2` = BTN0 (face/shoulders), `3` = BTN1 (triggers/select/start).

---

## MT62 Stream Format

### Header — 18 bytes, little-endian

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic `MT62` |
| 4 | 1 | Version (1) |
| 5 | 1 | FPS |
| 6 | 2 | Width (320) |
| 8 | 2 | Height (240) |
| 10 | 2 | Tile width (8) |
| 12 | 2 | Tile height (8) |
| 14 | 4 | Frame count |

### Frame — 18,848 bytes (fixed)

| Offset | Size | Content |
|--------|------|---------|
| 0 | 32 | `palette1` — base layer (Layer 1), 16 × RGB555, all opaque |
| 32 | 32 | `palette2` — overlay layer (Layer 2), 16 × RGB555, index 0 transparent |
| 64 | 8192 | `tiles2` — 256 tiles for overlay layer |
| 8256 | 8192 | `tiles1` — 256 tiles for base layer |
| 16448 | 1200 | `map2` — 40×30 tile IDs for overlay layer |
| 17648 | 1200 | `map1` — 40×30 tile IDs for base layer |

### RGB555 encoding

```
word = ((b >> 3) << 11) | ((g >> 3) << 6) | (r >> 3)
opaque bit = bit 5 of the high byte (0x0020)
```

The RP6502 VGA hardware treats index 0 of the overlay palette as transparent
when the opaque bit is clear.

### Hardware layer stack

```
Layer 0  (unused)
Layer 1  MIDDLE — base, always opaque
Layer 2  TOP    — overlay, index 0 transparent
```

---

## Key Engineering Findings

### Luma-based layer partition (critical)

The two 16-colour palettes are split by **luma rank**, not pixel count.
KMeans centroid luma values drift slowly between frames (especially with
warm-starting), so the assignment is stable.

Frequency-count ordering causes colours near rank 15/16 to swap palettes
every frame when their screen coverage differs by less than 1%, producing
alternating bands of noise across the top or bottom third of the image.
Sorting by luma eliminates this completely.

### Zero-tile pin in tile clustering

After clustering 8×8 tile patches, the tile with the lowest pixel sum is
swapped into slot 0 and zeroed out. Without this, KMeans can elect a
film-grain-contaminated almost-black tile as the representative for the darkest
cluster, stamping a fixed noise pattern over every dark cell in the frame.

### Double-buffered XRAM

The player streams each 18,848-byte frame directly from USB → XRAM into an
alternate buffer while the VGA hardware reads the previous buffer. On vsync,
config pointers are updated atomically. This eliminates mid-frame tears that
were visible with a single-buffer design.

### 2×2 pixel binning (V4 encoder)

Operating on a 160×120 image makes every 8×8 tile contain only a 4×4 grid of
distinct 2×2 blocks. The number of unique tile patterns drops dramatically,
tile KMeans converges faster, and the resulting tile assignments are stable
enough that temporal blend filtering is not needed at all.
