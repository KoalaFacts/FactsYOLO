/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#include "facts_yolo.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

_Static_assert(sizeof(fy_abi_info) == 84, "ABI info size");
_Static_assert(offsetof(fy_abi_info, build_id) == 20, "build id offset");
_Static_assert(sizeof(fy_detection) == 24, "fixed detection stride");
_Static_assert(sizeof(fy_capability) == 20, "fixed capability stride");

int main(void) {
    fy_abi_info out;
    unsigned char buffer[40];
    fy_diagnostic diag;
    out.struct_size = sizeof out;
    out.struct_version = FY_STRUCT_VERSION;
    out.required_flags = 0;
    memset(buffer, 0xA5, sizeof buffer);
    diag.struct_size = sizeof diag;
    diag.struct_version = FY_STRUCT_VERSION;
    diag.required_flags = 0;
    diag.data = (char *)buffer;
    diag.capacity = 8;
    assert(fy_abi_query(&out, &diag) == FY_OK);
    assert(out.abi_major == 1 && out.abi_minor == 0);
    assert(out.build_id[0] != 0 && out.build_id[63] == 0);
    assert(diag.required_bytes == 0 && buffer[0] == 0);
    out.struct_version = 2;
    out.abi_major = 77;
    assert(fy_abi_query(&out, &diag) == FY_UNSUPPORTED_ABI);
    assert(out.abi_major == 77);
    assert(diag.required_bytes == strlen("unsupported descriptor version"));
    assert(memcmp(buffer, "unsuppo", 7) == 0 && buffer[7] == 0);
    assert(buffer[8] == 0xA5);
    assert(fy_abi_query(NULL, NULL) == FY_INVALID_ARGUMENT);
#ifdef __linux__
    long page = sysconf(_SC_PAGESIZE);
    assert(page > 0);
    unsigned char *region = mmap(NULL, (size_t)page * 2, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(region != MAP_FAILED);
    assert(mprotect(region + page, (size_t)page, PROT_NONE) == 0);
    uint32_t *prefix = (uint32_t *)(region + page - 8);
    prefix[0] = 8;
    prefix[1] = 1;
    assert(fy_abi_query((fy_abi_info *)prefix, NULL) == FY_INVALID_ARGUMENT);
    out.struct_version = 1;
    assert(fy_abi_query(&out, (fy_diagnostic *)prefix) == FY_INVALID_ARGUMENT);
    assert(munmap(region, (size_t)page * 2) == 0);
#endif
    puts("C ABI query, diagnostic canaries and prefix guard-page checks passed");
    return 0;
}
