/* SPDX-License-Identifier: Apache-2.0 */
/* Standalone exact-path probe: no link-time dependency on either library. */
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[2], "normal") != 0 && strcmp(argv[2], "fixture") != 0)) return 3;
#ifdef _WIN32
    HMODULE library = LoadLibraryExA(argv[1], NULL,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (library == NULL) { puts("LOAD_FAILED"); return 2; }
    int query = GetProcAddress(library, "fy_abi_query") != NULL;
    int fixture = GetProcAddress(library, "fy_test_model_create") != NULL;
    int closed = FreeLibrary(library) != 0;
#else
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) { puts("LOAD_FAILED"); return 2; }
    int query = dlsym(library, "fy_abi_query") != NULL;
    int fixture = dlsym(library, "fy_test_model_create") != NULL;
    int closed = dlclose(library) == 0;
#endif
    if (!query || !closed || fixture != (strcmp(argv[2], "fixture") == 0)) return 4;
    puts("exact-path library capability isolation passed");
    return 0;
}
