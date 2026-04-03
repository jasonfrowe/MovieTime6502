/*
 * MovieTime6502_lb — ML62 low-bandwidth stream player
 *
 * Tile dictionaries are loaded ONCE at startup from TILES1.BIN and TILES2.BIN
 * on the USB mass-storage device.  Only palette + tilemap data (2,464 bytes
 * per frame) is streamed during playback, giving an ~8× bandwidth reduction
 * compared to the full MT62 player (18,848 bytes/frame).
 *
 * XRAM layout:
 *   0x0000 – 0x1FFF   tiles1  base    layer dictionary  (8,192 bytes, fixed)
 *   0x2000 – 0x3FFF   tiles2  overlay layer dictionary  (8,192 bytes, fixed)
 *   0x4000 – 0x499F   frame buffer 0  (2,464 bytes: pal1+pal2+map1+map2)
 *   0x49A0 – 0x533F   frame buffer 1  (2,464 bytes)
 *   0x5340 – 0x5353   tilemap1 VGA config (vga_mode2_config_t)
 *   0x5354 – 0x5367   tilemap2 VGA config (vga_mode2_config_t)
 *
 * Fixed tile scheme — tile index N:
 *   H = N >> 4   : colour applied to screen rows 0-3 of the tile
 *   L = N & 0x0F : colour applied to screen rows 4-7 of the tile
 *   tiles1 (base)    : even screen cols active; odd cols = index 0
 *   tiles2 (overlay) : odd screen cols active; even cols = index 0 (transparent)
 *
 * Files expected on USB root:
 *   TILES1.BIN — 8,192-byte base    layer tile dictionary
 *   TILES2.BIN — 8,192-byte overlay layer tile dictionary
 *   MOVIE.BIN  — ML62 stream (or filename supplied as argv[0/1])
 */

#include <rp6502.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include "constants_lb.h"

// Provide argv storage so the RP6502 C runtime can populate argc/argv.
void *argv_mem(size_t size)
{
    static unsigned char argv_storage[512];
    if (size > sizeof(argv_storage))
        return NULL;
    return argv_storage;
}

// ---------------------------------------------------------------------------
// VGA config addresses (set at init, tile_ptr never changes during playback)
// ---------------------------------------------------------------------------
static unsigned tilemap1_cfg;
static unsigned tilemap2_cfg;

// ---------------------------------------------------------------------------
// Vsync helpers
// ---------------------------------------------------------------------------
static const uint8_t cadence_24[CADENCE_LEN] = {2, 3, 2, 3, 2};
static uint8_t vsync_last;

static void wait_vsync(void)
{
    while (RIA.vsync == vsync_last)
        ;
    vsync_last = RIA.vsync;
}

static void wait_vsyncs(uint8_t n)
{
    uint8_t i;
    for (i = 0; i < n; i++)
        wait_vsync();
}

// ---------------------------------------------------------------------------
// Tile loading — reads TILES1.BIN and TILES2.BIN from USB into fixed XRAM
// locations.  Tile pointers in the VGA config structs never need to change.
// ---------------------------------------------------------------------------
static bool load_tile_dict(const char *filename, unsigned xram_base)
{
    int fd;
    int n;

    fd = open(filename, O_RDONLY);
    if (fd < 0) {
        printf("Cannot open %s: %d\n", filename, errno);
        return false;
    }

    n = read_xram(xram_base, TILE_DICT_BYTES, fd);
    close(fd);

    if (n != (int)TILE_DICT_BYTES) {
        printf("Short read %s: got %d\n", filename, n);
        return false;
    }

    printf("Loaded %s -> 0x%04X\n", filename, xram_base);
    return true;
}

// ---------------------------------------------------------------------------
// Graphics init — VGA mode-2 layers with tile_ptr pointing at the fixed
// XRAM tile dictionaries.  Palette and map pointers will be updated each
// frame; tile_ptr stays constant for the entire session.
// ---------------------------------------------------------------------------
static bool init_graphics(void)
{
    if (xreg_vga_canvas(1) < 0) {
        puts("canvas failed");
        return false;
    }

    // Layer 1 (MIDDLE) — base layer, always fully opaque.
    // tile_ptr = TILES1_XRAM_BASE (fixed for entire playback).
    // Palette and map pointers are initialised to buffer 0 and updated per frame.
    tilemap1_cfg = TILEMAP1_CONFIG_ADDR;
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, x_wrap,          false);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, y_wrap,          false);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, x_pos_px,        0);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, y_pos_px,        0);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, width_tiles,     WIDTH_TILES);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, height_tiles,    HEIGHT_TILES);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_palette_ptr,
                     (FRAME_BUF0_BASE + OFFSET_PAL_BASE));
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_tile_ptr,
                     TILES1_XRAM_BASE);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_data_ptr,
                     (FRAME_BUF0_BASE + OFFSET_MAP_BASE));
    if (xreg_vga_mode(2, 0x02, tilemap1_cfg, 1, 0, 0) < 0) {
        puts("mode1 failed");
        return false;
    }

    // Layer 2 (TOP) — overlay layer, palette index 0 is transparent.
    // tile_ptr = TILES2_XRAM_BASE (fixed for entire playback).
    tilemap2_cfg = TILEMAP2_CONFIG_ADDR;
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, x_wrap,          false);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, y_wrap,          false);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, x_pos_px,        0);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, y_pos_px,        0);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, width_tiles,     WIDTH_TILES);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, height_tiles,    HEIGHT_TILES);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_palette_ptr,
                     (FRAME_BUF0_BASE + OFFSET_PAL_OVERLAY));
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_tile_ptr,
                     TILES2_XRAM_BASE);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_data_ptr,
                     (FRAME_BUF0_BASE + OFFSET_MAP_OVERLAY));
    if (xreg_vga_mode(2, 0x02, tilemap2_cfg, 2, 0, 0) < 0) {
        puts("mode2 failed");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int      fd;
    int      i;
    int      n;
    uint8_t  header_fps;
    uint32_t frame_count;
    uint32_t frame_idx;
    uint8_t  cadence_pos;
    long     t_start, t_now;
    uint32_t frames_shown;
    uint32_t bytes_read;
    const char *filename;
    const char *arg0;
    size_t      arg0_len;

    printf("MovieTime6502_lb\n");

    printf("argc=%d\n", argc);
    for (i = 0; i < argc; i++) {
        printf("argv[%d]=%s\n", i, argv[i] ? argv[i] : "(null)");
    }

    // Accept movie filename from argv[1], argv[0] (if it ends in .BIN), or use default.
    filename = MOVIE_FILE_DEFAULT;
    if (argc > 1 && argv[1] != NULL && argv[1][0] != '\0') {
        filename = argv[1];
    } else if (argc > 0 && argv[0] != NULL && argv[0][0] != '\0') {
        arg0 = argv[0];
        arg0_len = strlen(arg0);
        if (arg0_len >= 4) {
            const char *ext = arg0 + (arg0_len - 4);
            if (ext[0] == '.' &&
                (ext[1] == 'B' || ext[1] == 'b') &&
                (ext[2] == 'I' || ext[2] == 'i') &&
                (ext[3] == 'N' || ext[3] == 'n'))
            {
                filename = arg0;
            }
        }
    }
    printf("Playing: %s\n", filename);

    // ---- Load fixed tile dictionaries into XRAM ----------------------------
    if (!load_tile_dict(TILES1_FILE, TILES1_XRAM_BASE)) {
        puts("Tile load failed");
        return 1;
    }
    if (!load_tile_dict(TILES2_FILE, TILES2_XRAM_BASE)) {
        puts("Tile load failed");
        return 1;
    }

    // ---- Init VGA -----------------------------------------------------------
    if (!init_graphics()) {
        puts("Graphics init failed");
        return 1;
    }

    // ---- Open the ML62 stream -----------------------------------------------
    fd = open(filename, O_RDONLY);
    if (fd < 0) {
        printf("Cannot open %s: %d\n", filename, errno);
        return 1;
    }

    // ---- Read and validate 18-byte header -----------------------------------
    //   [0..3]   magic    "ML62"
    //   [4]      version  1
    //   [5]      fps
    //   [6..7]   width    (informational)
    //   [8..9]   height   (informational)
    //   [10..11] tile_w   (informational)
    //   [12..13] tile_h   (informational)
    //   [14..17] frame_count  (little-endian uint32)
    {
        unsigned char hdr[HEADER_BYTES];
        n = read(fd, hdr, HEADER_BYTES);
        if (n != (int)HEADER_BYTES ||
            hdr[0] != 'M' || hdr[1] != 'L' || hdr[2] != '6' || hdr[3] != '2')
        {
            puts("Bad header (expected ML62)");
            close(fd);
            return 1;
        }
        header_fps  = hdr[5];
        frame_count = (uint32_t)hdr[14]
                | ((uint32_t)hdr[15] << 8)
                | ((uint32_t)hdr[16] << 16)
                | ((uint32_t)hdr[17] << 24);
        printf("FPS:%u  Frames:%lu\n", header_fps, (unsigned long)frame_count);
    }

    (void)header_fps; // only 24 fps cadence is hard-coded; see cadence_24

    // ---- Playback loop -------------------------------------------------------
    //
    // True double-buffering: read_xram fills 'read_buffer' while the GPU
    // renders from 'disp_buffer'.  After each vsync the buffers are swapped
    // and the VGA config structs updated (palette + map pointers only;
    // tile_ptr was set at init and is never touched again).
    //
    cadence_pos  = 0;
    frames_shown = 0;
    bytes_read   = 0;
    t_start      = clock();
    vsync_last   = RIA.vsync;

    unsigned read_buffer = FRAME_BUF0_BASE;
    unsigned disp_buffer = FRAME_BUF1_BASE;

    for (frame_idx = 0; frame_idx < frame_count; frame_idx++) {

        // Stream palette + maps directly from USB → XRAM frame buffer.
        // read_xram writes LB_FRAME_BYTES contiguous bytes starting at read_buffer,
        // exactly matching the buffer's OFFSET_PAL_BASE / OFFSET_MAP_BASE layout.
        n = read_xram(read_buffer, LB_FRAME_BYTES, fd);
        if (n != (int)LB_FRAME_BYTES)
            break;
        bytes_read += (uint32_t)n;

        // Wait for next vsync then flip buffers.
        wait_vsync();

        disp_buffer = read_buffer;
        read_buffer = (disp_buffer == FRAME_BUF0_BASE)
                      ? FRAME_BUF1_BASE
                      : FRAME_BUF0_BASE;

        // Update palette and map pointers to the new display buffer.
        // tile_ptr is NOT updated — it always points to the fixed dictionaries.
        xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_palette_ptr,
                         (disp_buffer + OFFSET_PAL_BASE));
        xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_data_ptr,
                         (disp_buffer + OFFSET_MAP_BASE));

        xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_palette_ptr,
                         (disp_buffer + OFFSET_PAL_OVERLAY));
        xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_data_ptr,
                         (disp_buffer + OFFSET_MAP_OVERLAY));

        // Hold for the remaining vsyncs in the current cadence slot.
        {
            uint8_t hold = cadence_24[cadence_pos];
            if (cadence_pos + 1 < CADENCE_LEN)
                cadence_pos++;
            else
                cadence_pos = 0;
            if (hold > 1)
                wait_vsyncs(hold - 1);
        }

        frames_shown++;

        // Periodic status (~every 2 seconds of wall time at 100 Hz clock).
        t_now = clock();
        if ((t_now - t_start) >= 200L) {
            long kb_s = (long)(bytes_read / 1024L) * 100L / (t_now - t_start);
            printf("F:%lu  %ld KB/s\n",
                   (unsigned long)frames_shown,
                   kb_s);
            t_start    = t_now;
            bytes_read = 0;
        }
    }

    close(fd);
    printf("Done. %lu frames shown.\n", (unsigned long)frames_shown);
    return 0;
}
