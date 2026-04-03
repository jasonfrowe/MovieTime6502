/*
 * input.c — Gamepad and keyboard input for MovieTime6502
 *
 * Reads RP6502 XRAM-mapped HID state each frame and exposes four
 * media-player actions with both edge (pressed) and level (held) queries.
 *
 * Mappings
 * --------
 * ACTION_PLAY_PAUSE  : Space / Enter            |  gamepad START
 * ACTION_STOP        : Escape / Q               |  gamepad SELECT
 * ACTION_FAST_FORWARD: Right Arrow / L          |  D-Pad Right / R1
 * ACTION_REWIND      : Left Arrow  / J          |  D-Pad Left  / L1
 */

#include <rp6502.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "input.h"
#include "usb_hid_keys.h"

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

uint8_t  keystates[KEYBOARD_BYTES];
gamepad_t gamepad[GAMEPAD_COUNT];

// Previous-frame raw state for edge detection
static uint8_t  prev_keystates[KEYBOARD_BYTES];
static gamepad_t prev_gamepad[GAMEPAD_COUNT];

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Returns true when action was NOT active last frame but IS active now.
static bool _raw_active(MediaAction action, const uint8_t *ks, const gamepad_t *gp)
{
    switch (action) {

    case ACTION_PLAY_PAUSE:
        if (ks[KEY_SPACE >> 3] & (1u << (KEY_SPACE & 7u))) return true;
        if (ks[KEY_ENTER >> 3] & (1u << (KEY_ENTER & 7u))) return true;
        if (gp[0].dpad & GP_CONNECTED) {
            if (gp[0].btn1 & GP_BTN_START) return true;
        }
        return false;

    case ACTION_STOP:
        if (ks[KEY_ESC >> 3] & (1u << (KEY_ESC & 7u))) return true;
        if (ks[KEY_Q   >> 3] & (1u << (KEY_Q   & 7u))) return true;
        if (gp[0].dpad & GP_CONNECTED) {
            if (gp[0].btn1 & GP_BTN_SELECT) return true;
        }
        return false;

    case ACTION_FAST_FORWARD:
        if (ks[KEY_RIGHT >> 3] & (1u << (KEY_RIGHT & 7u))) return true;
        if (ks[KEY_L     >> 3] & (1u << (KEY_L     & 7u))) return true;
        if (gp[0].dpad & GP_CONNECTED) {
            if (gp[0].dpad & GP_DPAD_RIGHT) return true;
            if (gp[0].btn0 & GP_BTN_R1)     return true;
        }
        return false;

    case ACTION_REWIND:
        if (ks[KEY_LEFT >> 3] & (1u << (KEY_LEFT & 7u))) return true;
        if (ks[KEY_J    >> 3] & (1u << (KEY_J    & 7u))) return true;
        if (gp[0].dpad & GP_CONNECTED) {
            if (gp[0].dpad & GP_DPAD_LEFT) return true;
            if (gp[0].btn0 & GP_BTN_L1)   return true;
        }
        return false;

    default:
        return false;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void init_input(void)
{
    memset(keystates,      0, sizeof(keystates));
    memset(prev_keystates, 0, sizeof(prev_keystates));
    memset(gamepad,      0, sizeof(gamepad));
    memset(prev_gamepad, 0, sizeof(prev_gamepad));
}

void poll_input(void)
{
    uint8_t i;

    // Save previous state for edge detection
    memcpy(prev_keystates, keystates, sizeof(keystates));
    memcpy(prev_gamepad,   gamepad,   sizeof(gamepad));

    // Read keyboard bitfield (256 keys packed 8 per byte)
    RIA.addr0 = KEYBOARD_INPUT;
    RIA.step0 = 1;
    for (i = 0; i < KEYBOARD_BYTES; i++) {
        keystates[i] = RIA.rw0;
    }

    // Read gamepad data (10 bytes × 4 pads)
    RIA.addr0 = GAMEPAD_INPUT;
    RIA.step0 = 1;
    for (i = 0; i < GAMEPAD_COUNT; i++) {
        gamepad[i].dpad   = RIA.rw0;
        gamepad[i].sticks = RIA.rw0;
        gamepad[i].btn0   = RIA.rw0;
        gamepad[i].btn1   = RIA.rw0;
        gamepad[i].lx     = (int8_t)RIA.rw0;
        gamepad[i].ly     = (int8_t)RIA.rw0;
        gamepad[i].rx     = (int8_t)RIA.rw0;
        gamepad[i].ry     = (int8_t)RIA.rw0;
        gamepad[i].l2     = RIA.rw0;
        gamepad[i].r2     = RIA.rw0;
    }
}

// Rising-edge: action is active now but was not active last frame.
bool action_pressed(MediaAction action)
{
    return _raw_active(action, keystates, gamepad) &&
           !_raw_active(action, prev_keystates, prev_gamepad);
}

// Level: action is currently active (held).
bool action_held(MediaAction action)
{
    return _raw_active(action, keystates, gamepad);
}
