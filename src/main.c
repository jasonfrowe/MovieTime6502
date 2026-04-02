#include <rp6502.h>
#include <stdio.h>
#include <stdbool.h>
#include "constants.h"
#include "palette.h"

unsigned TILEMAP1_CONFIG;
unsigned TILEMAP2_CONFIG;

static void init_graphics(void)
{

    // Select a 320x240 canvas
    if (xreg_vga_canvas(1) < 0) {
        puts("xreg_vga_canvas failed");
        return;
    }

    RIA.addr0 = PALETTE_ADDR1;
    RIA.step0 = 1;
    for (int i = 0; i < 16; i++) {
        RIA.rw0 = tile_palette[i] & 0xFF;
        RIA.rw0 = tile_palette[i] >> 8;
    }

    RIA.addr0 = PALETTE_ADDR2;
    RIA.step0 = 1;
    for (int i = 0; i < 16; i++) {
        RIA.rw0 = tile_palette[i] & 0xFF;
        RIA.rw0 = tile_palette[i] >> 8;
    }

    TILEMAP1_CONFIG = SPRITE_DATA_END;

    xram0_struct_set(TILEMAP1_CONFIG, vga_mode2_config_t, x_wrap, false);
    xram0_struct_set(TILEMAP1_CONFIG, vga_mode2_config_t, y_wrap, false);
    xram0_struct_set(TILEMAP1_CONFIG, vga_mode2_config_t, x_pos_px, 0);
    xram0_struct_set(TILEMAP1_CONFIG, vga_mode2_config_t, y_pos_px, 0);
    xram0_struct_set(TILEMAP1_CONFIG, vga_mode2_config_t, width_tiles,  WIDTH_TILES);
    xram0_struct_set(TILEMAP1_CONFIG, vga_mode2_config_t, height_tiles, HEIGHT_TILES);
    xram0_struct_set(TILEMAP1_CONFIG, vga_mode2_config_t, xram_data_ptr,    TILEMAP1_DATA); // tile ID grid
    xram0_struct_set(TILEMAP1_CONFIG, vga_mode2_config_t, xram_palette_ptr, PALETTE_ADDR1);
    xram0_struct_set(TILEMAP1_CONFIG, vga_mode2_config_t, xram_tile_ptr,    TILES1_DATA);  

    // Mode 2 args: MODE, OPTIONS, CONFIG, PLANE, BEGIN, END
    // OPTIONS: bit3=0 (8x8 tiles), bit[2:0]=2 (4-bit color index) => 0b0010 = 2
    // Plane 0 = background fill layer (behind sprite plane 1)
    if (xreg_vga_mode(2, 0x02, TILEMAP1_CONFIG, 2, 0, 0) < 0) {
        puts("xreg_vga_mode failed");
        return;
    }

    TILEMAP2_CONFIG = TILEMAP1_CONFIG + sizeof(vga_mode2_config_t);

    xram0_struct_set(TILEMAP2_CONFIG, vga_mode2_config_t, x_wrap, false);
    xram0_struct_set(TILEMAP2_CONFIG, vga_mode2_config_t, y_wrap, false);
    xram0_struct_set(TILEMAP2_CONFIG, vga_mode2_config_t, x_pos_px, 0);
    xram0_struct_set(TILEMAP2_CONFIG, vga_mode2_config_t, y_pos_px, 0);
    xram0_struct_set(TILEMAP2_CONFIG, vga_mode2_config_t, width_tiles,  WIDTH_TILES);
    xram0_struct_set(TILEMAP2_CONFIG, vga_mode2_config_t, height_tiles, HEIGHT_TILES);
    xram0_struct_set(TILEMAP2_CONFIG, vga_mode2_config_t, xram_data_ptr,    TILEMAP2_DATA); // tile ID grid
    xram0_struct_set(TILEMAP2_CONFIG, vga_mode2_config_t, xram_palette_ptr, PALETTE_ADDR2);
    xram0_struct_set(TILEMAP2_CONFIG, vga_mode2_config_t, xram_tile_ptr,    TILES2_DATA);  

    // Mode 2 args: MODE, OPTIONS, CONFIG, PLANE, BEGIN, END
    // OPTIONS: bit3=0 (8x8 tiles), bit[2:0]=2 (4-bit color index) => 0b0010 = 2
    // Plane 0 = background fill layer (behind sprite plane 1)
    if (xreg_vga_mode(2, 0x02, TILEMAP2_CONFIG, 1, 0, 0) < 0) {
        puts("xreg_vga_mode failed");
        return;
    }


}

#define SONG_HZ 60
uint8_t vsync_last = 0;
uint16_t timer_accumulator = 0;
bool music_enabled = true;

int main(void)
{
    init_graphics();

    while (true) {
        // Main game loop
        // 1. SYNC
        if (RIA.vsync == vsync_last) continue;
        vsync_last = RIA.vsync;

    }

    return 0;
}