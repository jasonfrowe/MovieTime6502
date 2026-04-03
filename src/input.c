/*
 * input.c — Gamepad and keyboard input for MovieTime6502
 *
 * Reads RP6502 XRAM-mapped HID state each frame and exposes four
 * media-player actions with both edge (pressed) and level (held) queries.
 *
 * Default mappings
 * ----------------
 * ACTION_PLAY_PAUSE  : Space / Enter            |  gamepad START
 * ACTION_STOP        : Escape / Q               |  gamepad SELECT
 * ACTION_FAST_FORWARD: Right Arrow / L          |  D-Pad Right / R1
 * ACTION_REWIND      : Left Arrow  / J          |  D-Pad Left  / L1
 *
 * Gamepad buttons can be remapped at runtime by placing JOYSTICK_CA.DAT
 * (preferred) or JOYSTICK.DAT on the USB drive.  File format:
 *   1 byte   num_mappings
 *   N × 3 bytes  { uint8_t action_id, uint8_t field, uint8_t mask }
 * where field is GP_FIELD_DPAD/STICKS/BTN0/BTN1 and action_id is a
 * MediaAction enum value.  The file replaces only the primary gamepad
 * binding for each listed action; keyboard bindings are unchanged.
 */

#include <rp6502.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include "input.h"
#include "usb_hid_keys.h"

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

uint8_t   keystates[KEYBOARD_BYTES];
gamepad_t gamepad[GAMEPAD_COUNT];

// Previous-frame raw state for edge detection
static uint8_t    prev_keystates[KEYBOARD_BYTES];
static gamepad_t  prev_gamepad[GAMEPAD_COUNT];

// ---------------------------------------------------------------------------
// Button mapping table — one entry per MediaAction
// ---------------------------------------------------------------------------
static ButtonMapping mappings[ACTION_COUNT];

static void set_default_mappings(void)
{
    memset(mappings, 0, sizeof(mappings));

    mappings[ACTION_PLAY_PAUSE].keyboard_key   = KEY_SPACE;
    mappings[ACTION_PLAY_PAUSE].keyboard_key2  = KEY_ENTER;
    mappings[ACTION_PLAY_PAUSE].gamepad_field  = GP_FIELD_BTN1;
    mappings[ACTION_PLAY_PAUSE].gamepad_mask   = GP_BTN_START;

    mappings[ACTION_STOP].keyboard_key   = KEY_ESC;
    mappings[ACTION_STOP].keyboard_key2  = KEY_Q;
    mappings[ACTION_STOP].gamepad_field  = GP_FIELD_BTN1;
    mappings[ACTION_STOP].gamepad_mask   = GP_BTN_SELECT;

    mappings[ACTION_FAST_FORWARD].keyboard_key   = KEY_RIGHT;
    mappings[ACTION_FAST_FORWARD].keyboard_key2  = KEY_L;
    mappings[ACTION_FAST_FORWARD].gamepad_field  = GP_FIELD_DPAD;
    mappings[ACTION_FAST_FORWARD].gamepad_mask   = GP_DPAD_RIGHT;
    mappings[ACTION_FAST_FORWARD].gamepad_field2 = GP_FIELD_BTN0;
    mappings[ACTION_FAST_FORWARD].gamepad_mask2  = GP_BTN_R1;

    mappings[ACTION_REWIND].keyboard_key   = KEY_LEFT;
    mappings[ACTION_REWIND].keyboard_key2  = KEY_J;
    mappings[ACTION_REWIND].gamepad_field  = GP_FIELD_DPAD;
    mappings[ACTION_REWIND].gamepad_mask   = GP_DPAD_LEFT;
    mappings[ACTION_REWIND].gamepad_field2 = GP_FIELD_BTN0;
    mappings[ACTION_REWIND].gamepad_mask2  = GP_BTN_L1;
}

// ---------------------------------------------------------------------------
// Joystick remapper file loader
// Tries JOYSTICK_CA.DAT first, falls back to JOYSTICK.DAT.
// Only the primary gamepad binding is replaced; keyboard keys are kept.
// ---------------------------------------------------------------------------
static void load_joystick_config(void)
{
    typedef struct {
        uint8_t action_id;
        uint8_t field;
        uint8_t mask;
    } JoystickMapping;

    int fd = open("JOYSTICK_CA.DAT", O_RDONLY);
    if (fd < 0)
        fd = open("JOYSTICK.DAT", O_RDONLY);
    if (fd < 0)
        return; // No config file present; keep defaults

    uint8_t num_mappings = 0;
    if (read(fd, &num_mappings, 1) != 1 || num_mappings == 0) {
        close(fd);
        return;
    }

    JoystickMapping file_map[ACTION_COUNT];
    if (num_mappings > ACTION_COUNT)
        num_mappings = ACTION_COUNT;

    int bytes = (int)num_mappings * (int)sizeof(JoystickMapping);
    if (read(fd, file_map, bytes) != bytes) {
        close(fd);
        return;
    }
    close(fd);

    uint8_t i;
    for (i = 0; i < num_mappings; i++) {
        if (file_map[i].action_id < ACTION_COUNT) {
            mappings[file_map[i].action_id].gamepad_field = file_map[i].field;
            mappings[file_map[i].action_id].gamepad_mask  = file_map[i].mask;
            // Clear secondary mapping when remapped so it won't override intent
            mappings[file_map[i].action_id].gamepad_mask2 = 0;
        }
    }
    puts("Joystick config loaded.");
}

// ---------------------------------------------------------------------------
// Internal: test raw button state using the mapping table
// ---------------------------------------------------------------------------
static uint8_t field_byte(const gamepad_t *gp, uint8_t field)
{
    switch (field) {
        case GP_FIELD_DPAD:   return gp->dpad;
        case GP_FIELD_STICKS: return gp->sticks;
        case GP_FIELD_BTN0:   return gp->btn0;
        case GP_FIELD_BTN1:   return gp->btn1;
        default:              return 0;
    }
}

static bool _raw_active(MediaAction action,
                         const uint8_t *ks, const gamepad_t *gp)
{
    if (action >= ACTION_COUNT) return false;
    const ButtonMapping *m = &mappings[action];

    // Keyboard — primary key
    if (m->keyboard_key && (ks[m->keyboard_key >> 3] & (1u << (m->keyboard_key & 7u))))
        return true;
    // Keyboard — secondary key
    if (m->keyboard_key2 && (ks[m->keyboard_key2 >> 3] & (1u << (m->keyboard_key2 & 7u))))
        return true;

    // Gamepad — only when a pad is connected
    if (!(gp[0].dpad & GP_CONNECTED))
        return false;

    // Primary gamepad binding
    if (m->gamepad_mask && (field_byte(&gp[0], m->gamepad_field) & m->gamepad_mask))
        return true;
    // Secondary gamepad binding
    if (m->gamepad_mask2 && (field_byte(&gp[0], m->gamepad_field2) & m->gamepad_mask2))
        return true;

    return false;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void init_input(void)
{
    // Register XRAM addresses with the RP6502 firmware so the Pico
    // actually writes keyboard and gamepad data into XRAM each frame.
    xregn(0, 0, 0, 1, KEYBOARD_INPUT);
    xregn(0, 0, 2, 1, GAMEPAD_INPUT);

    set_default_mappings();
    load_joystick_config();

    memset(keystates,      0, sizeof(keystates));
    memset(prev_keystates, 0, sizeof(prev_keystates));
    memset(gamepad,        0, sizeof(gamepad));
    memset(prev_gamepad,   0, sizeof(prev_gamepad));
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
    for (i = 0; i < KEYBOARD_BYTES; i++)
        keystates[i] = RIA.rw0;

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

// Rising-edge: active now but was NOT active last frame.
bool action_pressed(MediaAction action)
{
    return _raw_active(action, keystates, gamepad) &&
           !_raw_active(action, prev_keystates, prev_gamepad);
}

// Level: currently active (held).
bool action_held(MediaAction action)
{
    return _raw_active(action, keystates, gamepad);
}

