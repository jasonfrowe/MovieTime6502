#ifndef INPUT_H
#define INPUT_H

#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// RP6502 XRAM-mapped input addresses
// ---------------------------------------------------------------------------
#define GAMEPAD_INPUT   0xFF78U  // 40 bytes for 4 gamepads (10 bytes each)
#define KEYBOARD_INPUT  0xFFA0U  // 32 bytes keyboard bitfield (256 keys / 8)

// ---------------------------------------------------------------------------
// Gamepad hardware bit masks
// ---------------------------------------------------------------------------

// dpad byte
#define GP_DPAD_UP        0x01
#define GP_DPAD_DOWN      0x02
#define GP_DPAD_LEFT      0x04
#define GP_DPAD_RIGHT     0x08
#define GP_CONNECTED      0x80  // Set when a gamepad is plugged in

// sticks byte (digital thresholds reported by firmware)
#define GP_LSTICK_UP      0x01
#define GP_LSTICK_DOWN    0x02
#define GP_LSTICK_LEFT    0x04
#define GP_LSTICK_RIGHT   0x08
#define GP_RSTICK_UP      0x10
#define GP_RSTICK_DOWN    0x20
#define GP_RSTICK_LEFT    0x40
#define GP_RSTICK_RIGHT   0x80

// btn0 byte — face buttons and first shoulder pair
#define GP_BTN_A          0x01  // A / Cross
#define GP_BTN_B          0x02  // B / Circle
#define GP_BTN_C          0x04  // C / Right Paddle
#define GP_BTN_X          0x08  // X / Square
#define GP_BTN_Y          0x10  // Y / Triangle
#define GP_BTN_Z          0x20  // Z / Left Paddle
#define GP_BTN_L1         0x40
#define GP_BTN_R1         0x80

// btn1 byte — triggers and system buttons
#define GP_BTN_L2         0x01
#define GP_BTN_R2         0x02
#define GP_BTN_SELECT     0x04
#define GP_BTN_START      0x08
#define GP_BTN_HOME       0x10
#define GP_BTN_L3         0x20
#define GP_BTN_R3         0x40

// ---------------------------------------------------------------------------
// Keyboard: 32-byte bitfield; key(code) tests a single bit.
// ---------------------------------------------------------------------------
#define KEYBOARD_BYTES  32

extern uint8_t keystates[KEYBOARD_BYTES];

// Test whether USB HID key `code` is currently held.
#define key(code) (keystates[(code) >> 3] & (1u << ((code) & 7u)))

// ---------------------------------------------------------------------------
// Gamepad state (10 bytes per pad, matching RP6502 XRAM layout)
// ---------------------------------------------------------------------------
typedef struct {
    uint8_t dpad;    // D-pad directions + status bits
    uint8_t sticks;  // Digital stick thresholds
    uint8_t btn0;    // Face buttons / L1, R1
    uint8_t btn1;    // L2, R2, Select, Start, Home, L3, R3
    int8_t  lx;      // Left  stick X  (-128..127)
    int8_t  ly;      // Left  stick Y
    int8_t  rx;      // Right stick X
    int8_t  ry;      // Right stick Y
    uint8_t l2;      // Left  trigger (0-255)
    uint8_t r2;      // Right trigger
} gamepad_t;

#define GAMEPAD_COUNT 4

extern gamepad_t gamepad[GAMEPAD_COUNT];

// ---------------------------------------------------------------------------// Gamepad field selectors (match RP6502 gamepad_t byte order)
// ---------------------------------------------------------------------------
#define GP_FIELD_DPAD    0
#define GP_FIELD_STICKS  1
#define GP_FIELD_BTN0    2
#define GP_FIELD_BTN1    3

// ---------------------------------------------------------------------------
// Button mapping: one entry per action, for both keyboard and gamepad.
// Keyboard: two optional USB HID keycodes (key2 == 0 means unused).
// Gamepad:  primary field+mask, optional secondary field+mask (mask2==0=none).
// ---------------------------------------------------------------------------
typedef struct {
    uint8_t keyboard_key;   // USB HID keycode (primary)
    uint8_t keyboard_key2;  // USB HID keycode (secondary, 0 = none)
    uint8_t gamepad_field;  // GP_FIELD_* constant
    uint8_t gamepad_mask;   // Bit mask within that field (0 = unmapped)
    uint8_t gamepad_field2; // Secondary GP field
    uint8_t gamepad_mask2;  // Secondary bit mask (0 = none)
} ButtonMapping;

// ---------------------------------------------------------------------------// Media player actions
// ---------------------------------------------------------------------------
typedef enum {
    ACTION_PLAY_PAUSE,   // Space / Enter  |  START
    ACTION_STOP,         // Escape / Q     |  SELECT
    ACTION_FAST_FORWARD, // Right / L      |  D-Pad Right / R1
    ACTION_REWIND,       // Left  / J      |  D-Pad Left  / L1
    ACTION_COUNT
} MediaAction;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void init_input(void);
void poll_input(void);

// Returns true when action transitions from not-pressed → pressed (edge).
bool action_pressed(MediaAction action);

// Returns true while action is held (level).
bool action_held(MediaAction action);

#endif // INPUT_H
