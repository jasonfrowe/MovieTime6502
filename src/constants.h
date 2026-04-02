#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdint.h>

// ---------------------------------------------------------------------------
// Screen / tile geometry
// ---------------------------------------------------------------------------
#define SCREEN_WIDTH    320
#define SCREEN_HEIGHT   240
#define TILE_W          8
#define TILE_H          8
#define WIDTH_TILES     40   // SCREEN_WIDTH  / TILE_W
#define HEIGHT_TILES    30   // SCREEN_HEIGHT / TILE_H

// ---------------------------------------------------------------------------
// XRAM layout  (all addresses within the 64 KB XRAM window)
//
//   0x0000 – 0x04AF   map1      (40×30 = 1200 bytes, overlay layer)
//   0x04B0 – 0x095F   map2      (40×30 = 1200 bytes, base layer)
//   0x0960 – 0x295F   tiles1    (256 tiles × 32 bytes, overlay layer)
//   0x2960 – 0x495F   tiles2    (256 tiles × 32 bytes, base layer)
//   0x4960 – 0x496F   tilemap1 config  (vga_mode2_config_t, 20 bytes)
//   0x4974 – 0x4987   tilemap2 config  (vga_mode2_config_t, 20 bytes)
//   0xFC00 – 0xFC1F   palette1  (16 × 2 bytes RGB555, overlay)
//   0xFC20 – 0xFC3F   palette2  (16 × 2 bytes RGB555, base)
// ---------------------------------------------------------------------------

// Tile-map data (40×30 byte ID arrays)
#define TILEMAP1_DATA       0x0000U
#define TILEMAP1_DATA_SIZE  0x04B0U   // 1200 bytes

#define TILEMAP2_DATA       (TILEMAP1_DATA + TILEMAP1_DATA_SIZE)
#define TILEMAP2_DATA_SIZE  0x04B0U   // 1200 bytes

// Tile-image data (256 tiles × 32 bytes each)
#define TILES1_DATA         (TILEMAP2_DATA + TILEMAP2_DATA_SIZE)
#define TILES1_DATA_SIZE    0x2000U   // 8192 bytes

#define TILES2_DATA         (TILES1_DATA + TILES1_DATA_SIZE)
#define TILES2_DATA_SIZE    0x2000U   // 8192 bytes

// End of streaming data — config structs placed here
#define XRAM_DATA_END       (TILES2_DATA + TILES2_DATA_SIZE)   // 0x4960

// vga_mode2_config_t is 20 bytes (sizeof verified at build time)
#define TILEMAP1_CONFIG_ADDR    XRAM_DATA_END
#define TILEMAP2_CONFIG_ADDR    (TILEMAP1_CONFIG_ADDR + 20U)

// Palettes (fixed high-XRAM slots used by the VGA hardware)
#define PALETTE_ADDR1   0xFC00U  // overlay layer — 16 × RGB555 = 32 bytes
#define PALETTE_ADDR2   0xFC20U  // base layer    — 16 × RGB555 = 32 bytes
#define PALETTE_SIZE    0x0020U  // 32 bytes per palette

// ---------------------------------------------------------------------------
// MT62 movie stream format constants
// ---------------------------------------------------------------------------
#define MOVIE_FILE              "MOVIE.BIN"

// Per-frame section sizes (bytes)
#define FRAME_PALETTE1_BYTES    32U    // palette1 (overlay)
#define FRAME_PALETTE2_BYTES    32U    // palette2 (base)
#define FRAME_TILES2_BYTES      8192U  // 256 tiles, base layer
#define FRAME_TILES1_BYTES      8192U  // 256 tiles, overlay layer
#define FRAME_MAP2_BYTES        1200U  // 40×30 tile IDs, base layer
#define FRAME_MAP1_BYTES        1200U  // 40×30 tile IDs, overlay layer

// Total bytes per frame
#define FRAME_BYTES  (FRAME_PALETTE1_BYTES + FRAME_PALETTE2_BYTES + \
                      FRAME_TILES2_BYTES   + FRAME_TILES1_BYTES   + \
                      FRAME_MAP2_BYTES     + FRAME_MAP1_BYTES)
// = 18,848

// File header size
#define HEADER_BYTES    18U

// Cadence table length (24 fps on 60 Hz: 2-3-2-3-2 pattern)
#define CADENCE_LEN     5U

#endif // CONSTANTS_H