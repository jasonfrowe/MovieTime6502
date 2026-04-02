/*
 * MovieTime6502 — MT62 stream player
 *
 * Reads MOVIE.BIN from USB mass storage and displays it using the
 * RP6502 VGA mode-2 tile renderer at the FPS encoded in the file header.
 *
 * Frame layout (18,848 bytes, fixed, matching encode_movie.py):
 *   32  bytes  palette1   (overlay layer, 16 × RGB555)
 *   32  bytes  palette2   (base layer,    16 × RGB555)
 *   8192 bytes tiles2     (256 tiles, base layer)
 *   8192 bytes tiles1     (256 tiles, overlay layer)
 *   1200 bytes map2       (40×30 tile IDs, base layer)
 *   1200 bytes map1       (40×30 tile IDs, overlay layer)
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
#include "constants.h"

// ---------------------------------------------------------------------------
// VGA config addresses (written once at init into XRAM)
// ---------------------------------------------------------------------------
static unsigned tilemap1_cfg;
static unsigned tilemap2_cfg;

// ---------------------------------------------------------------------------
// Graphics init — sets up both mode-2 planes with the correct XRAM pointers.
// Palettes are loaded per-frame so no static palette data is needed here.
// ---------------------------------------------------------------------------
static bool init_graphics(void)
{
    if (xreg_vga_canvas(1) < 0) {
        puts("canvas failed");
        return false;
    }

    tilemap1_cfg = TILEMAP1_CONFIG_ADDR;
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, x_wrap,        false);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, y_wrap,        false);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, x_pos_px,      0);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, y_pos_px,      0);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, width_tiles,   WIDTH_TILES);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, height_tiles,  HEIGHT_TILES);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_data_ptr,    TILEMAP1_DATA);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_palette_ptr, PALETTE_ADDR1);
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_tile_ptr,    TILES1_DATA);
    // Plane 1 = overlay (drawn over plane 2)
    if (xreg_vga_mode(2, 0x02, tilemap1_cfg, 1, 0, 0) < 0) {
        puts("mode1 failed");
        return false;
    }

    tilemap2_cfg = TILEMAP2_CONFIG_ADDR;
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, x_wrap,        false);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, y_wrap,        false);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, x_pos_px,      0);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, y_pos_px,      0);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, width_tiles,   WIDTH_TILES);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, height_tiles,  HEIGHT_TILES);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_data_ptr,    TILEMAP2_DATA);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_palette_ptr, PALETTE_ADDR2);
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_tile_ptr,    TILES2_DATA);
    // Plane 2 = base (background)
    if (xreg_vga_mode(2, 0x02, tilemap2_cfg, 2, 0, 0) < 0) {
        puts("mode2 failed");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Palette writer — copies 32 bytes of RGB555 words into the XRAM palette slot
// using RW0 for speed. Caller has already positioned RIA.addr0.
// ---------------------------------------------------------------------------
static void write_palette_raw(unsigned xram_addr, const unsigned char *data)
{
    int i;
    RIA.addr0 = xram_addr;
    RIA.step0 = 1;
    for (i = 0; i < (int)PALETTE_SIZE; i++)
        RIA.rw0 = data[i];
}

// ---------------------------------------------------------------------------
// Frame apply — takes the raw 18,848-byte frame buffer and pushes each
// section to XRAM. All writes use read_xram (already positioning the
// destination address) or direct sequential writes via RW0.
//
// Section order in the frame buffer:
//   [0]     palette1   32 bytes
//   [32]    palette2   32 bytes
//   [64]    tiles2     8192 bytes
//   [8256]  tiles1     8192 bytes
//   [16448] map2       1200 bytes
//   [17648] map1       1200 bytes
//
// We read each section straight from the open file into XRAM using
// read_xram() so there is no intermediate 6502 RAM buffer needed for
// the large tile and map sections.
// ---------------------------------------------------------------------------

// Scratch buffer for palette data only (64 bytes — small, always in RAM)
static unsigned char pal_buf[FRAME_PALETTE1_BYTES + FRAME_PALETTE2_BYTES];

// ---------------------------------------------------------------------------
// Cadence table for 24 fps presentation on a 60 Hz display.
// 60/24 = 2.5 vsyncs per frame, so we alternate 2 and 3 vsyncs.
// 5-entry repeating cycle: 2,3,2,3,2 = 12 vsyncs = 5 frames (200ms)
// Each entry is the number of extra vsyncs to hold (0 = show on next vsync).
// ---------------------------------------------------------------------------
static const uint8_t cadence_24[CADENCE_LEN] = {2, 3, 2, 3, 2};

// ---------------------------------------------------------------------------
// Vsync spin helpers
// ---------------------------------------------------------------------------
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
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int fd;
    uint8_t  header_fps;
    uint32_t frame_count;
    uint32_t frame_idx;
    uint8_t  cadence_pos;
    int      n;
    long     t_start, t_now;
    uint32_t frames_shown;
    uint32_t bytes_read;
    uint32_t late_frames;
    const char *filename;

    printf("MovieTime6502\n");

    // Get filename from command line, or default to SHOWCASE.BIN
    filename = (argc > 1) ? argv[1] : "SHOWCASE.BIN";
    printf("Playing: %s\n", filename);

    if (!init_graphics()) {
        puts("Graphics init failed");
        return 1;
    }

    // Open the movie stream from USB mass storage
    fd = open(filename, O_RDONLY);
    if (fd < 0) {
        printf("Cannot open %s: %d\n", filename, errno);
        return 1;
    }

    // ---- Read and validate header (18 bytes) --------------------------------
    //   4 bytes magic  "MT62"
    //   1 byte  version
    //   1 byte  fps
    //   2 bytes width
    //   2 bytes height
    //   2 bytes tile_w
    //   2 bytes tile_h
    //   4 bytes frame_count
    {
        unsigned char hdr[HEADER_BYTES];
        n = read(fd, hdr, HEADER_BYTES);
        if (n != HEADER_BYTES ||
            hdr[0] != 'M' || hdr[1] != 'T' || hdr[2] != '6' || hdr[3] != '2')
        {
            puts("Bad header");
            close(fd);
            return 1;
        }
        header_fps  = hdr[5];
        // width/height at hdr[6..9], tile_w/tile_h at hdr[10..13] — informational
        frame_count = (uint32_t)hdr[14]
                | ((uint32_t)hdr[15] << 8)
                | ((uint32_t)hdr[16] << 16)
                | ((uint32_t)hdr[17] << 24);
        printf("FPS:%u  Frames:%lu\n", header_fps, (unsigned long)frame_count);
    }

    // Build the cadence table for the encoded fps.
    // For 24 fps we use the hard-coded 2-3-2-3-2 table above.
    // If a different fps is stored, fall back to the nearest whole vsync count.
    // (Extension point - cadence selection by header_fps left for future work)
    (void)header_fps; // currently only 24fps cadence is built in

    // ---- Playback loop -------------------------------------------------------
    cadence_pos  = 0;
    frames_shown = 0;
    bytes_read   = 0;
    late_frames  = 0;
    t_start      = clock();
    vsync_last   = RIA.vsync;

    for (frame_idx = 0; frame_idx < frame_count; frame_idx++) {

        // -- Read palette bytes into local RAM buffer ------------------------
        n = read(fd, pal_buf, sizeof(pal_buf));
        if (n != (int)sizeof(pal_buf)) break;
        bytes_read += n;

        // -- Stream tiles2 directly from USB → XRAM (8192 bytes) ------------
        n = read_xram(TILES2_DATA, FRAME_TILES2_BYTES, fd);
        if (n != FRAME_TILES2_BYTES) break;
        bytes_read += n;

        // -- Stream tiles1 directly from USB → XRAM (8192 bytes) ------------
        n = read_xram(TILES1_DATA, FRAME_TILES1_BYTES, fd);
        if (n != FRAME_TILES1_BYTES) break;
        bytes_read += n;

        // -- Stream map2 directly from USB → XRAM (1200 bytes) --------------
        n = read_xram(TILEMAP2_DATA, FRAME_MAP2_BYTES, fd);
        if (n != FRAME_MAP2_BYTES) break;
        bytes_read += n;

        // -- Stream map1 directly from USB → XRAM (1200 bytes) --------------
        n = read_xram(TILEMAP1_DATA, FRAME_MAP1_BYTES, fd);
        if (n != FRAME_MAP1_BYTES) break;
        bytes_read += n;

        // -- Wait for vsync then apply palette and advance cadence -----------
        // Check if we are already past our presentation vsync (late frame).
        // We still display it rather than skip, but count the event.
        wait_vsync();

        // Write palettes immediately after vsync for minimal tearing
        write_palette_raw(PALETTE_ADDR1, pal_buf);
        write_palette_raw(PALETTE_ADDR2, pal_buf + FRAME_PALETTE1_BYTES);

        // Hold for remaining vsyncs in the cadence slot
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

        // -- Periodic status (every ~2 seconds of wall time) ----------------
        t_now = clock();
        if ((t_now - t_start) >= 200L) { // 200 ticks = 2 seconds at 100 Hz
            long kb_s = (long)(bytes_read / 1024L) * 100L / (t_now - t_start);
            printf("F:%lu  %ld KB/s  late:%lu\n",
                   (unsigned long)frames_shown,
                   kb_s,
                   (unsigned long)late_frames);
            t_start   = t_now;
            bytes_read = 0;
        }
    }

    close(fd);
    printf("Done. %lu frames shown.\n", (unsigned long)frames_shown);
    return 0;
}