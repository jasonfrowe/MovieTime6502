#ifndef CONSTANTS_H
#define CONSTANTS_H

// Screen dimensions
#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 240

// Tile dimensions
#define TILE_W 8
#define TILE_H 8

// Main Map configuration
#define WIDTH_TILES 40
#define HEIGHT_TILES 30

// Sprite data configuration
#define SPRITE_DATA_START        0x0000U // Starting address in XRAM for sprite data

#define TILEMAP1_DATA           (SPRITE_DATA_START) // Tilemap 1 data
#define TILEMAP1_DATA_SIZE      0x04B0U             // 320*240/(8*8) = 1200 bytes

#define TILEMAP2_DATA           (TILEMAP1_DATA + TILEMAP1_DATA_SIZE) // Tilemap 2 data
#define TILEMAP2_DATA_SIZE      0x04B0U             // 1200 bytes

#define TILES1_DATA              (TILEMAP2_DATA + TILEMAP2_DATA_SIZE) // Tileset 1 data
#define TILES1_DATA_SIZE         0x2000U             // 8192 bytes

#define TILES2_DATA              (TILES1_DATA + TILES1_DATA_SIZE) // Tileset 2 data
#define TILES2_DATA_SIZE         0x2000U             // 8192 bytes

#define SPRITE_DATA_END         (TILES2_DATA + TILES2_DATA_SIZE) // End of sprite data

// Input buffers
// #define GAMEPAD_INPUT   0xFF78  // 40 bytes for 4 gamepads
// #define KEYBOARD_INPUT  0xFFA0  // 32 bytes keyboard bitfield

// Palette and sound
#define PALETTE_ADDR1   0xFC00  // 16-color palette (32 bytes, 0xFC00-0xFC1F)
#define PALETTE_SIZE    0x0020  // 32 bytes

#define PALETTE_ADDR2   0xFC20  // Starting address for palette data (32 bytes, 0xFC20-0xFC3F)
#define PALETTE_SIZE2   0x0020  // 32 bytes

// #define OPL_ADDR        0xFE00  // OPL2 register page (256 bytes, must be page-aligned)
// #define OPL_SIZE        0x0100

#endif // CONSTANTS_H