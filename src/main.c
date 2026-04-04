/*
 * MovieTime6502 — MT62 stream player
 *
 * Reads MOVIE.BIN from USB mass storage and displays it using the
 * RP6502 VGA mode-2 tile renderer at the FPS encoded in the file header.
 *
 * Frame layout (18,848 bytes, fixed, matching encode_movie.py):
 *   32   bytes  palette1   (base layer,    16 × RGB555)
 *   32   bytes  palette2   (overlay layer, 16 × RGB555)
 *   8192 bytes  tiles2     (256 tiles, overlay layer)
 *   8192 bytes  tiles1     (256 tiles, base layer)
 *   1200 bytes  map2       (40×30 tile IDs, overlay layer)
 *   1200 bytes  map1       (40×30 tile IDs, base layer)
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
#include "input.h"

// Provide argv storage so RP6502 C runtime can populate argc/argv.
// Without this hook, argc can be 0 even when launch arguments are supplied.
void *argv_mem(size_t size)
{
    static unsigned char argv_storage[512];
    if (size > sizeof(argv_storage))
        return NULL;
    return argv_storage;
}

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
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_palette_ptr, (BUFFER0_BASE + OFFSET_PAL_BASE));
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_tile_ptr,    (BUFFER0_BASE + OFFSET_TILES_BASE));
    xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_data_ptr,    (BUFFER0_BASE + OFFSET_MAP_BASE));
    // Layer 1 (MIDDLE) = base background; always fully opaque.
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
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_palette_ptr, (BUFFER0_BASE + OFFSET_PAL_OVERLAY));
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_tile_ptr,    (BUFFER0_BASE + OFFSET_TILES_OVERLAY));
    xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_data_ptr,    (BUFFER0_BASE + OFFSET_MAP_OVERLAY));
    // Layer 2 (TOP) = overlay; drawn over Layer 1; palette index 0 is transparent.
    if (xreg_vga_mode(2, 0x02, tilemap2_cfg, 2, 0, 0) < 0) {
        puts("mode2 failed");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Debug layer view controls (compile-time)
//   DEBUG_VIEW_NORMAL       : show both layers (default)
//   DEBUG_VIEW_BASE_ONLY    : hide top overlay layer
//   DEBUG_VIEW_OVERLAY_ONLY : hide middle base layer (shows overlay on black)
// ---------------------------------------------------------------------------
#define DEBUG_VIEW_NORMAL        0
#define DEBUG_VIEW_BASE_ONLY     1
#define DEBUG_VIEW_OVERLAY_ONLY  2

#ifndef DEBUG_VIEW_MODE
#define DEBUG_VIEW_MODE DEBUG_VIEW_NORMAL
#endif

// ---------------------------------------------------------------------------
// Vsync helpers — 3:2 pulldown for 24 fps on 60 Hz.
// Even film frames are held 2 display vsyncs, odd frames 3 (avg 2.5).
// vsync_target is the absolute RIA.vsync counter value at which to swap.
// ---------------------------------------------------------------------------
static uint8_t vsync_target;

static void wait_for_target(void)
{
    while ((int8_t)(RIA.vsync - vsync_target) < 0)
        ;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int fd;
    int i;
    uint8_t  header_fps;
    uint32_t frame_count;
    uint32_t frame_idx;
    int      n;
    long     t_start, t_now;
    uint32_t frames_shown;
    uint32_t bytes_read;
    uint32_t late_frames;
    const char *filename;
    const char *arg0;
    size_t      arg0_len;

    printf("MovieTime6502\n");

    // Debug: inspect argv provided by launcher/monitor.
    printf("argc=%d\n", argc);
    for (i = 0; i < argc; i++) {
        printf("argv[%d]=%s\n", i, argv[i] ? argv[i] : "(null)");
    }

    // Get filename from command line, or default to SHOWCASE.BIN.
    // Some launch paths may provide only one argv entry, so also accept
    // argv[0] when it looks like a .BIN filename.
    filename = "SHOWCASE.BIN";
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

#if DEBUG_VIEW_MODE == DEBUG_VIEW_BASE_ONLY
    puts("Debug view: BASE ONLY");
#elif DEBUG_VIEW_MODE == DEBUG_VIEW_OVERLAY_ONLY
    puts("Debug view: OVERLAY ONLY");
#else
    puts("Debug view: NORMAL");
#endif

    init_input();

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
    frames_shown = 0;
    bytes_read   = 0;
    late_frames  = 0;
    t_start      = clock();
    vsync_target = RIA.vsync + 1;

    bool paused = false;

    unsigned read_buffer = BUFFER0_BASE;
    unsigned disp_buffer = BUFFER1_BASE;

    for (frame_idx = 0; frame_idx < frame_count; frame_idx++) {

        // -- Poll input at the start of every frame -------------------------
        poll_input();

        // STOP — exit playback immediately
        if (action_pressed(ACTION_STOP)) {
            puts("Stopped.");
            break;
        }

        // PLAY/PAUSE toggle
        if (action_pressed(ACTION_PLAY_PAUSE)) {
            paused = !paused;
            puts(paused ? "Paused." : "Playing.");
        }

        // FAST FORWARD — skip ahead SKIP_FRAMES frames
        if (action_held(ACTION_FAST_FORWARD) && !paused) {
            long skip_bytes = (long)FRAME_BYTES * SKIP_FRAMES;
            if (lseek(fd, skip_bytes, SEEK_CUR) < 0) {
                // Reached or passed end — let the read below handle termination
                lseek(fd, 0, SEEK_END);
            } else {
                frame_idx += SKIP_FRAMES;
                if (frame_idx >= frame_count)
                    frame_idx = frame_count - 1;
            }
        }

        // REWIND — seek back SKIP_FRAMES frames (clamped to start)
        if (action_held(ACTION_REWIND) && !paused) {
            long skip_bytes = (long)FRAME_BYTES * (SKIP_FRAMES + 1);
            long cur = lseek(fd, 0, SEEK_CUR);
            long target = cur - skip_bytes;
            long data_start = (long)HEADER_BYTES;
            if (target < data_start) target = data_start;
            lseek(fd, target, SEEK_SET);
            // Recalculate frame_idx from file position
            frame_idx = (uint32_t)((target - data_start) / FRAME_BYTES);
        }

        // Pause loop — keep displaying the last frame, still polling input
        while (paused) {
            vsync_target = RIA.vsync + 1;
            wait_for_target();
            poll_input();
            if (action_pressed(ACTION_PLAY_PAUSE)) {
                paused = false;
                puts("Playing.");
            }
            if (action_pressed(ACTION_STOP)) {
                paused = false;
                frame_idx = frame_count; // signal loop termination after while
                break;
            }
        }
        if (frame_idx >= frame_count) break;

        // -- Stream entire frame directly from USB → XRAM (18,848 bytes) -----
        n = read_xram(read_buffer, FRAME_BYTES, fd);
        if (n != FRAME_BYTES) break;
        bytes_read += n;

        // -- Wait for target vsync then swap buffers ------------------------
        // Late frames may cause a scanline of garbage
        disp_buffer = read_buffer;
        read_buffer = (disp_buffer == BUFFER0_BASE) ? BUFFER1_BASE : BUFFER0_BASE;
        if ((int8_t)(RIA.vsync - vsync_target) >= 0) 
            late_frames++;
        wait_for_target();

        // Apply new pointers immediately to live config structs
        xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_palette_ptr, (disp_buffer + OFFSET_PAL_BASE));
        xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_tile_ptr,    (disp_buffer + OFFSET_TILES_BASE));
        xram0_struct_set(tilemap1_cfg, vga_mode2_config_t, xram_data_ptr,    (disp_buffer + OFFSET_MAP_BASE));

        xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_palette_ptr, (disp_buffer + OFFSET_PAL_OVERLAY));
        xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_tile_ptr,    (disp_buffer + OFFSET_TILES_OVERLAY));
        xram0_struct_set(tilemap2_cfg, vga_mode2_config_t, xram_data_ptr,    (disp_buffer + OFFSET_MAP_OVERLAY));

        // Advance target: even frames 2 vsyncs, odd frames 3 (3:2 pulldown = 24 fps)
        vsync_target = RIA.vsync + ((frame_idx & 1) ? 3 : 2);

    #if DEBUG_VIEW_MODE == DEBUG_VIEW_BASE_ONLY
        // Force top overlay layer palette fully transparent.
        {
            int j;
            RIA.addr0 = disp_buffer + OFFSET_PAL_OVERLAY;
            RIA.step0 = 1;
            for (j = 0; j < 32; j++) RIA.rw0 = 0x00;
        }
    #elif DEBUG_VIEW_MODE == DEBUG_VIEW_OVERLAY_ONLY
        // Force middle base layer palette to opaque black.
        {
            int j;
            RIA.addr0 = disp_buffer + OFFSET_PAL_BASE;
            RIA.step0 = 1;
            for (j = 0; j < 16; j++) {
                RIA.rw0 = 0x20; // LSB
                RIA.rw0 = 0x00; // MSB
            }
        }
    #endif


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
