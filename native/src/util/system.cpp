#include "system.hpp"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

namespace ps {

uint64_t physicalMemory() {
#if defined(_WIN32)
    MEMORYSTATUSEX s;
    s.dwLength = sizeof(s);
    return GlobalMemoryStatusEx(&s) ? uint64_t(s.ullTotalPhys) : 0;
#elif defined(__APPLE__)
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    return sysctlbyname("hw.memsize", &bytes, &len, nullptr, 0) == 0 ? bytes : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES), size = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && size > 0 ? uint64_t(pages) * uint64_t(size) : 0;
#endif
}

}  // namespace ps
