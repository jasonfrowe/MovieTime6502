"""
Generate the fixed ML62 tile dictionaries and write images/tiles1.bin + images/tiles2.bin.

Tile index N encodes:
  H = N >> 4   top-half colour  (screen rows 0-3, binned rows 0-1)
  L = N & 0xF  bottom-half colour (screen rows 4-7, binned rows 2-3)

Base layer   tiles1.bin: active on even screen columns (0,2,4,6); odd cols = 0.
Overlay layer tiles2.bin: active on odd  screen columns (1,3,5,7); even cols = 0.

8x8 tile pixel arrays:
  base:    row 0-3: [H,0,H,0,H,0,H,0]   row 4-7: [L,0,L,0,L,0,L,0]
  overlay: row 0-3: [0,H,0,H,0,H,0,H]   row 4-7: [0,L,0,L,0,L,0,L]

VGA mode-2 packed format (32 bytes/tile):
  8 rows x 4 bytes; each byte = (lo_nibble) | (hi_nibble << 4)
  lo = pixel at even screen col pair, hi = pixel at odd screen col pair.
"""
import os
import numpy as np

TILE_H, TILE_W, NUM_TILES = 8, 8, 256


def encode_tile(pixels: np.ndarray) -> bytes:
    out = bytearray(32)
    for row in range(TILE_H):
        for col in range(TILE_W // 2):
            lo = int(pixels[row, col * 2]) & 0xF
            hi = int(pixels[row, col * 2 + 1]) & 0xF
            out[row * 4 + col] = lo | (hi << 4)
    return bytes(out)


tiles1 = bytearray()
tiles2 = bytearray()

for N in range(NUM_TILES):
    H = (N >> 4) & 0xF
    L = N & 0xF

    p1 = np.zeros((TILE_H, TILE_W), dtype=np.uint8)
    p1[:4, 0::2] = H
    p1[4:, 0::2] = L

    p2 = np.zeros((TILE_H, TILE_W), dtype=np.uint8)
    p2[:4, 1::2] = H
    p2[4:, 1::2] = L

    tiles1 += encode_tile(p1)
    tiles2 += encode_tile(p2)

os.makedirs("images", exist_ok=True)
with open("images/tiles1.bin", "wb") as f:
    f.write(tiles1)
with open("images/tiles2.bin", "wb") as f:
    f.write(tiles2)

assert len(tiles1) == NUM_TILES * 32 == 8192
assert tiles1[:32] == bytes(32), "tile 0 must be all-zero"
t = tiles1[0xFF * 32: 0xFF * 32 + 32]
assert t[:4] == bytes([0x0F, 0x0F, 0x0F, 0x0F]), f"tile 0xFF row0 wrong: {t[:4].hex()}"

print(f"tiles1.bin: {len(tiles1)} bytes")
print(f"tiles2.bin: {len(tiles2)} bytes")
print("Spot checks passed.")
