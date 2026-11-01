/* CrispASR 0.8.48's explicit CUDA preference otherwise falls back to CPU.
 * Parakeet/Canary call init_best only after preferred CUDA initialization fails.
 * Keep CPU helper backends available; reject an all-CPU primary backend.
 * This is deliberately tied to the pinned runtime, not a general ggml shim.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *(*init_fn)(void);
typedef const char *(*name_fn)(void *);

/* Do not preload this guard into any subprocess the speech engine starts. */
__attribute__((constructor)) static void scope_preload(void) {
    unsetenv("LD_PRELOAD");
}

void *ggml_backend_init_best(void) {
    init_fn init = (init_fn)dlsym(RTLD_NEXT, "ggml_backend_init_best");
    name_fn name = (name_fn)dlsym(RTLD_NEXT, "ggml_backend_name");
    if (!init || !name) {
        fputs("SpeechDock: incompatible CrispASR CUDA guard ABI; refusing inference\n", stderr);
        exit(78);
    }
    void *backend = init();
    if (!backend || strncmp(name(backend), "CUDA", 4) != 0) {
        fputs("SpeechDock: CUDA 12 GPU backend unavailable; CPU fallback is disabled\n", stderr);
        exit(78);
    }
    return backend;
}
