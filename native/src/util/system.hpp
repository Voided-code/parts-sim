// Facts about the machine the app runs on.
#pragma once

#include <cstdint>

namespace ps {

/** Installed RAM in bytes (0 when unknown). */
uint64_t physicalMemory();

}  // namespace ps
