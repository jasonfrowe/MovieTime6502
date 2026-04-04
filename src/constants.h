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
// MovieTime6502 uses full double buffering. 
// Each frame buffer exactly matches the 18,848 byte MT62 file frame layout:
//   [0]     palette1   32 bytes   (Base Layer)
//   [32]    palette2   32 bytes   (Overlay Layer)
//   [64]    tiles2     8192 bytes (Overlay Layer)
//   [8256]  tiles1     8192 bytes (Base Layer)
//   [16448] map2       1200 bytes (Overlay Layer)
//   [17648] map1       1200 bytes (Base Layer)
//
//   0x0000 – 0x499F   Buffer 0
//   0x49A0 – 0x933F   Buffer 1
//   0x9340 – 0x9353   tilemap1 config  (vga_mode2_config_t, 20 bytes)
//   0x9354 – 0x9367   tilemap2 config  (vga_mode2_config_t, 20 bytes)
// ---------------------------------------------------------------------------

#define BUFFER0_BASE        0x0000U
#define BUFFER1_BASE        0x49A0U

// Offsets matching the contiguous binary frame chunk
#define OFFSET_PAL_BASE      0U
#define OFFSET_PAL_OVERLAY   32U
#define OFFSET_TILES_OVERLAY 64U
#define OFFSET_TILES_BASE    8256U
#define OFFSET_MAP_OVERLAY   16448U
#define OFFSET_MAP_BASE      17648U

#define TILEMAP1_CONFIG_ADDR 0x9340U
#define TILEMAP2_CONFIG_ADDR 0x9354U

#define PALETTE_SIZE         32U

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

// Total bytes per frame — MT62 v1 (two separate 256-tile blocks, 18,848 bytes)
#define FRAME_BYTES  (FRAME_PALETTE1_BYTES + FRAME_PALETTE2_BYTES + \
                      FRAME_TILES2_BYTES   + FRAME_TILES1_BYTES   + \
                      FRAME_MAP2_BYTES     + FRAME_MAP1_BYTES)
// = 18,848

// Total bytes per frame — MT62 v2 / V3C (combined tile block, 10,656 bytes)
// Disk layout: pal1(32) + pal2(32) + combined_tiles(8192) + map2(1200) + map1(1200)
// combined_tiles[i] = (base_tiles[i] & 0x0F) | (overlay_tiles[i] & 0xF0)
// The Pico splits combined_tiles into separate XRAM tile slots via opcode 0x2F.
#define FRAME_BYTES_V3C  (FRAME_PALETTE1_BYTES + FRAME_PALETTE2_BYTES + \
                          8192U + FRAME_MAP2_BYTES + FRAME_MAP1_BYTES)
// = 10,656

// MT62 file format version bytes
#define MT62_VERSION_V1   1U  // two separate tile blocks (18,848 bytes/frame)
#define MT62_VERSION_V3C  2U  // combined tile block, Pico-split (10,656 bytes/frame)

// File header size
#define HEADER_BYTES    18U

// Number of frames to skip/rewind per fast-forward / rewind hold.
// At 24 fps, 24 = 1 second jump per input poll cycle.
#define SKIP_FRAMES     24U

#endif // CONSTANTS_H