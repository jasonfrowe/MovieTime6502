#ifndef CONSTANTS_LB_H
#define CONSTANTS_LB_H

#include <stdint.h>

// ---------------------------------------------------------------------------
// Screen / tile geometry (same as MT62 player)
// ---------------------------------------------------------------------------
#define SCREEN_WIDTH    320
#define SCREEN_HEIGHT   240
#define TILE_W          8
#define TILE_H          8
#define WIDTH_TILES     40   // SCREEN_WIDTH  / TILE_W
#define HEIGHT_TILES    30   // SCREEN_HEIGHT / TILE_H

// ---------------------------------------------------------------------------
// XRAM layout for MovieTime6502_lb  (ML62 Low-Bandwidth player)
//
// Unlike the MT62 player, tile dictionaries are fixed and loaded ONCE at
// startup from TILES1.BIN and TILES2.BIN.  Only palette + tilemap data
// (2,464 bytes/frame) is streamed per frame, giving ~8× bandwidth reduction.
//
// Fixed tile dictionaries (written once from TILES1.BIN / TILES2.BIN):
//   0x0000 – 0x1FFF  tiles1  base layer    (8,192 bytes, mathematically fixed)
//   0x2000 – 0x3FFF  tiles2  overlay layer (8,192 bytes, mathematically fixed)
//
// Double-buffered per-frame data (palette + maps only):
//   0x4000 – 0x499F  Frame buffer 0  (2,464 bytes)
//   0x49A0 – 0x533F  Frame buffer 1  (2,464 bytes)
//
// Per-buffer offsets (matching the ML62 binary frame layout exactly):
//   [0]      32 bytes  palette1  (base layer,    Layer 1)
//   [32]     32 bytes  palette2  (overlay layer, Layer 2, index 0 transparent)
//   [64]   1200 bytes  map1      (40×30 tile IDs, base layer)
//   [1264] 1200 bytes  map2      (40×30 tile IDs, overlay layer)
//   Total: 2,464 bytes
//
// VGA mode-2 config structs (written once at init; tile_ptr never changes):
//   0x5340 – 0x5353  tilemap1 config  (vga_mode2_config_t, 20 bytes)
//   0x5354 – 0x5367  tilemap2 config  (vga_mode2_config_t, 20 bytes)
// ---------------------------------------------------------------------------

// Fixed tile dictionary bases (tile_ptr in mode-2 config; set once, never updated)
#define TILES1_XRAM_BASE     0x0000U   // base    layer tile dictionary
#define TILES2_XRAM_BASE     0x2000U   // overlay layer tile dictionary
#define TILE_DICT_BYTES      8192U     // 256 tiles × 32 bytes/tile

// Double-buffered per-frame payloads
#define FRAME_BUF0_BASE      0x4000U
#define FRAME_BUF1_BASE      0x49A0U   // 0x4000 + 0x9A0 (2,464)

// Within each frame buffer (offsets matching binary stream layout)
#define OFFSET_PAL_BASE      0U        // palette1  — Layer 1 (base,    32 bytes)
#define OFFSET_PAL_OVERLAY   32U       // palette2  — Layer 2 (overlay, 32 bytes)
#define OFFSET_MAP_BASE      64U       // map1      — Layer 1 (1,200 bytes)
#define OFFSET_MAP_OVERLAY   1264U     // map2      — Layer 2 (1,200 bytes)

// VGA mode-2 config struct locations
#define TILEMAP1_CONFIG_ADDR 0x5340U
#define TILEMAP2_CONFIG_ADDR 0x5354U

// ---------------------------------------------------------------------------
// ML62 movie stream format constants
// ---------------------------------------------------------------------------
#define MOVIE_FILE_DEFAULT      "MOVIE.BIN"
#define TILES1_FILE             "ROM:tiles1.bin"
#define TILES2_FILE             "ROM:tiles2.bin"

// Header size (identical to MT62 header)
#define HEADER_BYTES            18U

// Per-frame section sizes
#define LB_FRAME_PAL1_BYTES     32U    // palette1 (base layer)
#define LB_FRAME_PAL2_BYTES     32U    // palette2 (overlay layer)
#define LB_FRAME_MAP1_BYTES     1200U  // 40×30 tile IDs, base layer
#define LB_FRAME_MAP2_BYTES     1200U  // 40×30 tile IDs, overlay layer

// Total bytes per frame in the ML62 stream
#define LB_FRAME_BYTES  (LB_FRAME_PAL1_BYTES + LB_FRAME_PAL2_BYTES + \
                         LB_FRAME_MAP1_BYTES  + LB_FRAME_MAP2_BYTES)
// = 2,464

// Cadence table length (24 fps on 60 Hz: 2-3-2-3-2 pattern)
#define CADENCE_LEN     5U

#endif // CONSTANTS_LB_H
