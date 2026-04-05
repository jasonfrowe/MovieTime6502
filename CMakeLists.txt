cmake_minimum_required(VERSION 3.18)

add_subdirectory(tools)

set(LLVM_MOS_PLATFORM rp6502)
# Prefer a locally installed SDK under external/llvm-mos.
if(EXISTS "${CMAKE_SOURCE_DIR}/external/llvm-mos")
    list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/llvm-mos")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/external/llvm-mos-sdk")
    # Fallback to source tree config; requires mos compilers available in PATH.
    list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/llvm-mos-sdk")
endif()
find_package(llvm-mos-sdk REQUIRED)

project(MovieTime6502 C CXX ASM)

# ---------------------------------------------------------------------------
# MovieTime6502 — MT62 USB stream player (stock firmware)
# Plays MT62 version 1 files (18,848 bytes/frame, two tile blocks).
# Runs on unmodified RP6502 firmware.
# ---------------------------------------------------------------------------

add_executable(MovieTime6502)
rp6502_asset(MovieTime6502 help src/help.txt)

rp6502_executable(MovieTime6502
    DATA file
    RESET file
)

target_sources(MovieTime6502 PRIVATE
    src/main.c
    src/input.c
)

# ---------------------------------------------------------------------------
# MovieTime6502_v3c — MT62 USB stream player (custom firmware required)
# Plays MT62 version 1 AND version 2 (V3C) files.
# V3C: 10,656 bytes/frame via combined tile block split by Pico opcode 0x2F.
# Requires RP6502 firmware with read_xram_split (opcode 0x2F) built in.
# ---------------------------------------------------------------------------

add_executable(MovieTime6502_v3c)
rp6502_asset(MovieTime6502_v3c help src/help.txt)

rp6502_executable(MovieTime6502_v3c
    DATA file
    RESET file
)

target_sources(MovieTime6502_v3c PRIVATE
    src/main.c
    src/input.c
    src/read_xram_split.c
)

target_compile_definitions(MovieTime6502_v3c PRIVATE
    ENABLE_V3C=1
)

# ---------------------------------------------------------------------------
# MovieTime6502_lb — ML62 low-bandwidth USB stream player
# Tile dictionaries (tiles1.bin, tiles2.bin) are embedded in the ROM.
# Only palette+map data (2,464 bytes/frame) is streamed from USB,
# giving ~8× bandwidth reduction vs the full MT62 player.
# ---------------------------------------------------------------------------

add_executable(MovieTime6502_lb)
rp6502_asset(MovieTime6502_lb help lb/help_lb.txt)
rp6502_asset(MovieTime6502_lb tiles1.bin images/tiles1.bin)
rp6502_asset(MovieTime6502_lb tiles2.bin images/tiles2.bin)

rp6502_executable(MovieTime6502_lb
    DATA file
    RESET file
)

target_sources(MovieTime6502_lb PRIVATE
    lb/MovieTime6502_lb.c
)
