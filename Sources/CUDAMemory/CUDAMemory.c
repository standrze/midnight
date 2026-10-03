#include "CUDAMemory.h"
#include <stdio.h>

#if defined(__linux__) && defined(MIDNIGHT_CUDA)
#include <cuda.h>
#include <dlfcn.h>
#include <string.h>

/* Load driver APIs without introducing CUDA linkage in CPU-only builds.
 * The legacy advice/prefetch entry points intentionally use their pre-CUDA-13
 * ABI, retained by the CUDA driver; the _v2 entry points have different types. */
struct driver {
    void *library;
    CUcontext context;
    int pushed;
    CUresult (*init)(unsigned int);
    CUresult (*count)(int *);
    CUresult (*attribute)(int *, CUdevice_attribute, CUdevice);
    CUresult (*retain)(CUcontext *, CUdevice);
    CUresult (*release)(CUdevice);
    CUresult (*push)(CUcontext);
    CUresult (*pop)(CUcontext *);
    CUresult (*info)(size_t *, size_t *);
    CUresult (*sync)(void);
    CUresult (*pointer_attribute)(void *, CUpointer_attribute, CUdeviceptr);
    CUresult (*advise)(CUdeviceptr, size_t, CUmem_advise, CUdevice);
    CUresult (*prefetch)(CUdeviceptr, size_t, CUdevice, CUstream);
    CUresult (*error_string)(CUresult, const char **);
};

static int fail(struct driver *d, CUresult result, const char *operation,
                char *error, size_t capacity) {
    const char *detail = "CUDA driver failure";
    if (d->error_string) d->error_string(result, &detail);
    snprintf(error, capacity, "%s: %s (%d)", operation, detail ? detail : "unknown", (int)result);
    return -1;
}

static void close_driver(struct driver *d) {
    if (d->pushed) {
        CUcontext previous;
        d->pop(&previous);
    }
    if (d->context) d->release(0);
    if (d->library) dlclose(d->library);
}

static int open_driver(struct driver *d, char *error, size_t capacity) {
    memset(d, 0, sizeof(*d));
    d->library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!d->library) {
        snprintf(error, capacity, "CUDA driver library libcuda.so.1 is unavailable");
        return -1;
    }
#define LOAD(field, symbol) do { \
    *(void **)(&d->field) = dlsym(d->library, symbol); \
    if (!d->field) { \
        snprintf(error, capacity, "CUDA driver is missing %s", symbol); \
        close_driver(d); return -1; \
    } \
} while (0)
    LOAD(init, "cuInit");
    LOAD(count, "cuDeviceGetCount");
    LOAD(attribute, "cuDeviceGetAttribute");
    LOAD(retain, "cuDevicePrimaryCtxRetain");
    LOAD(release, "cuDevicePrimaryCtxRelease_v2");
    LOAD(push, "cuCtxPushCurrent_v2");
    LOAD(pop, "cuCtxPopCurrent_v2");
    LOAD(info, "cuMemGetInfo_v2");
    LOAD(sync, "cuCtxSynchronize");
    LOAD(pointer_attribute, "cuPointerGetAttribute");
    LOAD(advise, "cuMemAdvise");
    LOAD(prefetch, "cuMemPrefetchAsync");
    LOAD(error_string, "cuGetErrorString");
#undef LOAD
    CUresult result = d->init(0);
    if (result == CUDA_SUCCESS) result = d->retain(&d->context, 0);
    if (result == CUDA_SUCCESS) {
        result = d->push(d->context);
        d->pushed = result == CUDA_SUCCESS;
    }
    if (result != CUDA_SUCCESS) {
        fail(d, result, "initializing CUDA memory query", error, capacity);
        close_driver(d);
        return -1;
    }
    return 0;
}

int midnight_cuda_memory_snapshot(uint64_t *free_bytes, uint64_t *total_bytes,
                                  int *managed, char *error, size_t capacity) {
    struct driver d;
    if (open_driver(&d, error, capacity)) return -1;
    size_t free_value = 0, total_value = 0;
    CUresult result = d.info(&free_value, &total_value);
    int count = 0;
    if (result == CUDA_SUCCESS) result = d.count(&count);
    *managed = count > 0;
    /* Match MLX's allocator: it uses managed memory only when every visible
     * GPU permits concurrent managed access. Logical zero honors CUDA_VISIBLE_DEVICES. */
    for (int i = 0; result == CUDA_SUCCESS && i < count; ++i) {
        int value = 0;
        result = d.attribute(&value, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, i);
        if (!value) *managed = 0;
    }
    if (result == CUDA_SUCCESS) {
        *free_bytes = free_value;
        *total_bytes = total_value;
    } else {
        fail(&d, result, "querying CUDA memory", error, capacity);
    }
    close_driver(&d);
    return result == CUDA_SUCCESS ? 0 : -1;
}

static int place(void *pointer, size_t bytes, int host, int reset,
                 char *error, size_t capacity) {
    struct driver d;
    if (open_driver(&d, error, capacity)) return -1;
    CUdeviceptr address = (CUdeviceptr)(uintptr_t)pointer;
    unsigned int managed = 0;
    CUresult result = d.pointer_attribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, address);
    if (result == CUDA_SUCCESS && !managed) result = CUDA_ERROR_NOT_SUPPORTED;
    /* Weight placement happens outside active model execution. Synchronize before
     * advice changes, prefetch, and release, including failure rollback. */
    if (result == CUDA_SUCCESS) result = d.sync();
    if (result == CUDA_SUCCESS) result = d.advise(address, bytes, CU_MEM_ADVISE_UNSET_READ_MOSTLY, 0);
    if (result == CUDA_SUCCESS) result = d.advise(address, bytes,
        reset ? CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION : CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
        host ? CU_DEVICE_CPU : 0);
    if (result == CUDA_SUCCESS) result = d.advise(address, bytes,
        reset ? CU_MEM_ADVISE_UNSET_ACCESSED_BY : CU_MEM_ADVISE_SET_ACCESSED_BY, 0);
    if (!reset && result == CUDA_SUCCESS) result = d.prefetch(address, bytes, host ? CU_DEVICE_CPU : 0, NULL);
    if (result == CUDA_SUCCESS) result = d.sync();
    if (result != CUDA_SUCCESS) fail(&d, result, reset ? "resetting CUDA weight placement" : "placing CUDA weights", error, capacity);
    close_driver(&d);
    return result == CUDA_SUCCESS ? 0 : -1;
}

int midnight_cuda_place_weights(void *pointer, size_t bytes, int host, char *error, size_t capacity) {
    return place(pointer, bytes, host, 0, error, capacity);
}
int midnight_cuda_reset_weights(void *pointer, size_t bytes, char *error, size_t capacity) {
    return place(pointer, bytes, 0, 1, error, capacity);
}
#else
static int unavailable(char *error, size_t capacity) {
    snprintf(error, capacity, "CUDA memory operations require a Linux CUDA build");
    return -1;
}
int midnight_cuda_memory_snapshot(uint64_t *f, uint64_t *t, int *m, char *e, size_t n) {
    (void)f; (void)t; (void)m;
    return unavailable(e, n);
}
int midnight_cuda_place_weights(void *p, size_t b, int h, char *e, size_t n) {
    (void)p; (void)b; (void)h;
    return unavailable(e, n);
}
int midnight_cuda_reset_weights(void *p, size_t b, char *e, size_t n) {
    (void)p; (void)b;
    return unavailable(e, n);
}
#endif
