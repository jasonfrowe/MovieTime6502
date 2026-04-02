# MovieTime6502

MovieTime6502 is an RP6502 demo that streams a pre-encoded tile movie from USB
mass storage and plays it with VGA mode 2 dual tile layers.

The runtime binary expects a file named MOVIE.BIN on the USB drive.

## Project Layout

- src/main.c: MT62 movie player runtime (reads MOVIE.BIN from USB)
- src/constants.h: XRAM layout and frame format constants
- tools/encode_movie.py: offline encoder and stream verifier

## Build

Initialize the vendored SDK submodule once:

git submodule update --init --recursive

Build and install the SDK locally (one-time, creates external/llvm-mos):

cmake -S external/llvm-mos-sdk -B external/llvm-mos-sdk/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/external/llvm-mos
cmake --build external/llvm-mos-sdk/build --target install -j4

1. Configure and build with CMake (or build from VS Code CMake Tools).
2. The build output includes MovieTime6502.rp6502 in build/target-native.

Typical command line build:

cmake -S . -B build/target-native
cmake --build build/target-native --target MovieTime6502

## Encoder Dependencies

Install Python dependencies once:

pip install opencv-python numpy scikit-learn Pillow

## Encode a Movie Clip

Important: encode_movie.py uses subcommands. Use encode for creating MOVIE.BIN
and verify for decoding inspection frames.

Example command for your Metropolis clip:

python tools/encode_movie.py encode \
  --input Sprites/Metropolis_1927.mp4 \
  --start "01:23:00" \
  --end "01:26:00" \
  --output MOVIE.BIN

The encoder scales to fill 320x240 and center-crops the excess width. It does
not letterbox.

Optional flags:

- --fps 24
- --debug-dir debug_frames
- --debug-every 24

## Verify the Encoded Stream

Generate a few reconstructed PNG frames to check quality:

python tools/encode_movie.py verify MOVIE.BIN --debug-dir debug_verify --frames 10

## Procedural Showcase Stream

If you want a synthetic demo instead of encoded film footage, generate a faux
3D tunnel showcase stream with rotating palettes and a transparent HUD layer:

python tools/generate_showcase.py --output SHOWCASE.BIN --seconds 20 --fps 24

To run it through the current player ROM, rename or copy the generated file to
MOVIE.BIN on the USB drive.

## Run on Hardware

1. Copy MOVIE.BIN to the root of the USB mass storage device.
2. Upload and run MovieTime6502.rp6502.
3. The player prints periodic throughput stats during playback.

## MT62 Stream Format

Header size: 18 bytes (magic, version, fps, resolution, tile size, frame count)

Frame size: 18,848 bytes

- palette1: 32 bytes
- palette2: 32 bytes
- tiles2: 8192 bytes
- tiles1: 8192 bytes
- map2: 1200 bytes
- map1: 1200 bytes

All format constants are mirrored in src/constants.h and tools/encode_movie.py.
