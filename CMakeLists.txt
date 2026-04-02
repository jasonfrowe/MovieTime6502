cmake_minimum_required(VERSION 3.18)

add_subdirectory(tools)

set(LLVM_MOS_PLATFORM rp6502)
find_package(llvm-mos-sdk REQUIRED)

project(MovieTime6502 C CXX ASM)

# ---------------------------------------------------------------------------
# MovieTime6502 — MT62 USB stream player
# MOVIE.BIN is read at runtime from USB mass storage; no assets are embedded.
# ---------------------------------------------------------------------------

add_executable(MovieTime6502)
rp6502_asset(MovieTime6502 help src/help.txt)

rp6502_executable(MovieTime6502
    DATA file
    RESET file
)

target_sources(MovieTime6502 PRIVATE
    src/main.c
)
