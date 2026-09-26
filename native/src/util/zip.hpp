// Reading ZIP archives (3MF) and raw/zlib DEFLATE streams (SolidWorks blocks), via zlib.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ps {

/** Inflate raw DEFLATE (windowBits -15) or zlib-wrapped data; empty on failure. */
std::vector<uint8_t> inflateRaw(const uint8_t* data, size_t size, size_t expected);
std::vector<uint8_t> inflateZlib(const uint8_t* data, size_t size, size_t expected);
uint32_t crc32(const uint8_t* data, size_t size);

/** All files of a ZIP archive by path (stored and deflated entries). Throws on a corrupt archive. */
std::map<std::string, std::vector<uint8_t>> unzip(const std::vector<uint8_t>& archive);

}  // namespace ps
