/*
 * read_xram_split — RIA opcode 0x2F
 *
 * Reads `count` bytes from fildes and splits each byte into two XRAM regions:
 *   xram[base_dst + i] = byte & 0x0F   (base layer,    lo nibble)
 *   xram[ov_dst   + i] = byte & 0xF0   (overlay layer, hi nibble)
 *
 * The split is performed on the Pico at 125 MHz, avoiding the 32,768
 * RIA register accesses that doing it on the 65C02 @ 8 MHz would require.
 *
 * Returns the number of file bytes consumed (= count on success), or -1.
 */

#include <rp6502.h>

#define RIA_OP_READ_XRAM_SPLIT 0x2F

int read_xram_split(unsigned base_dst, unsigned ov_dst, unsigned count, int fildes)
{
    ria_push_int(base_dst);
    ria_push_int(ov_dst);
    ria_push_int(count);
    ria_set_ax(fildes);
    return ria_call_int(RIA_OP_READ_XRAM_SPLIT);
}
