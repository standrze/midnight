// Observe a benchmark child's physical footprint using the macOS kernel ledger.
// Build before timing: xcrun clang -O2 Scripts/macos-process-footprint.c -o /tmp/probe
// Usage: probe PID INTERVAL_MS > memory.jsonl
// Lifetime maxima are observed while the process exists, not certified at exit.
#include <errno.h>
#include <inttypes.h>
#include <libproc.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static int positive_integer(const char *text, long maximum, long *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno || end == text || *end || parsed <= 0 || parsed > maximum) return 0;
    *value = parsed;
    return 1;
}

int main(int argc, char **argv) {
    long pid, interval;
    if (argc != 3 || !positive_integer(argv[1], 2147483647, &pid) ||
        !positive_integer(argv[2], 10000, &interval) || interval < 10) {
        fprintf(stderr, "Usage: footprint PID INTERVAL_MS (10..10000)\n");
        return 2;
    }
    struct timespec delay = {.tv_sec = interval / 1000, .tv_nsec = (interval % 1000) * 1000000};
    uint64_t samples = 0;
    while (1) {
        struct rusage_info_v4 usage = {0};
        errno = 0;
        int status = proc_pid_rusage((int)pid, RUSAGE_INFO_V4, (rusage_info_t *)&usage);
        if (status != 0) {
            int code = errno;
            printf("{\"event\":\"probe_stopped\",\"pid\":%ld,\"samples\":%" PRIu64
                   ",\"errno\":%d,\"reason\":\"process_unavailable\"}\n", pid, samples, code);
            fflush(stdout);
            return samples && code == ESRCH ? 0 : 1;
        }
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        printf("{\"event\":\"sample\",\"pid\":%ld,\"monotonic_seconds\":%.9f,"
               "\"physical_footprint_bytes\":%" PRIu64 ",\"resident_size_bytes\":%" PRIu64
               ",\"wired_size_bytes\":%" PRIu64 ",\"observed_lifetime_max_physical_footprint_bytes\":%" PRIu64 "}\n",
               pid, (double)now.tv_sec + (double)now.tv_nsec / 1e9,
               usage.ri_phys_footprint, usage.ri_resident_size, usage.ri_wired_size,
               usage.ri_lifetime_max_phys_footprint);
        fflush(stdout);
        samples++;
        nanosleep(&delay, NULL);
    }
}
