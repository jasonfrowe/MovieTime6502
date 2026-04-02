cmake_minimum_required(VERSION 3.18)

add_subdirectory(tools)

set(LLVM_MOS_PLATFORM rp6502)
find_package(llvm-mos-sdk REQUIRED)

project(MovieTime6502 C CXX ASM)

# Define the option (Default is ON/Native) 
option(USE_NATIVE_OPL2 "Use the RIA native OPL2 support" ON)

if(USE_NATIVE_OPL2)
    add_definitions(-DUSE_NATIVE_OPL2)
    message(STATUS "Targeting: Native RIA OPL2")
else()
    message(STATUS "Targeting: FPGA TinyFPGA Sound Card")
endif()

add_executable(MovieTime6502)
rp6502_asset(MovieTime6502 0x10000 images/tiles1.bin)
rp6502_asset(MovieTime6502 0x104B0 images/tiles2.bin)
rp6502_asset(MovieTime6502 help src/help.txt)

rp6502_executable(MovieTime6502 
    DATA file 
    RESET file
)

target_sources(MovieTime6502 PRIVATE
    src/main.c
)
