/* Launch only our colocated CrispASR, with a mandatory, already-open guard.
 * LD_PRELOAD alone ignores missing libraries; preflight dlopen and an inherited
 * /proc/self/fd reference make missing/unloadable/replaced guards fail closed.
 * CUDA library search paths are deliberately left unchanged.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void fail(const char *message) {
    fprintf(stderr, "SpeechDock: %s; refusing CrispASR inference\n", message);
    exit(78);
}

int main(int argc, char **argv) {
    char path[PATH_MAX], guard[PATH_MAX], fd_path[64];
    ssize_t length = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (length <= 0 || length >= (ssize_t)sizeof(path) - 1)
        fail("cannot resolve CUDA launcher");
    path[length] = '\0';
    char *slash = strrchr(path, '/');
    if (!slash) fail("invalid CUDA launcher path");
    *slash = '\0';
    if (snprintf(guard, sizeof(guard), "%s/cuda-required.so", path) >= (int)sizeof(guard))
        fail("CUDA guard path is too long");
    int fd = open(guard, O_RDONLY | O_NOFOLLOW);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) || !S_ISREG(st.st_mode))
        fail("CUDA guard is missing or unsafe");
    snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", fd);
    void *handle = dlopen(fd_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle || !dlsym(handle, "ggml_backend_init_best"))
        fail("CUDA guard cannot load on this system");
    dlclose(handle);
    if (setenv("LD_PRELOAD", fd_path, 1))
        fail("cannot activate required CUDA guard");
    if ((size_t)length + sizeof("/crispasr") >= sizeof(path))
        fail("CrispASR path is too long");
    strcat(path, "/crispasr");
    /* argv is already terminated and all caller arguments pass through unchanged. */
    (void)argc;
    argv[0] = path;
    execv(path, argv);
    fail("CrispASR executable or its system libraries are unavailable");
}
