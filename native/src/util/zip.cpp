#include "zip.hpp"

#include <stdexcept>
#include <zlib.h>

namespace ps {

static std::vector<uint8_t> inflateWith(const uint8_t* data, size_t size, size_t expected, int windowBits) {
    std::vector<uint8_t> out(expected ? expected : size * 4 + 64);
    z_stream s{};
    if (inflateInit2(&s, windowBits) != Z_OK) return {};
    s.next_in = const_cast<Bytef*>(data);
    s.avail_in = uInt(size);
    int rc;
    do {
        if (s.total_out >= out.size()) {
            if (expected) break;  // more than promised
            out.resize(out.size() * 2);
        }
        s.next_out = out.data() + s.total_out;
        s.avail_out = uInt(out.size() - s.total_out);
        rc = inflate(&s, Z_NO_FLUSH);
    } while (rc == Z_OK);
    const size_t produced = s.total_out;
    inflateEnd(&s);
    if (rc != Z_STREAM_END) return {};
    out.resize(produced);
    return out;
}

std::vector<uint8_t> inflateRaw(const uint8_t* data, size_t size, size_t expected) { return inflateWith(data, size, expected, -15); }
std::vector<uint8_t> inflateZlib(const uint8_t* data, size_t size, size_t expected) { return inflateWith(data, size, expected, 15); }

uint32_t crc32(const uint8_t* data, size_t size) { return uint32_t(::crc32(0L, data, uInt(size))); }

static uint32_t u32(const std::vector<uint8_t>& b, size_t at) {
    if (at + 4 > b.size()) throw std::runtime_error("The archive is truncated.");
    return b[at] | (b[at + 1] << 8) | (b[at + 2] << 16) | (uint32_t(b[at + 3]) << 24);
}
static uint16_t u16(const std::vector<uint8_t>& b, size_t at) {
    if (at + 2 > b.size()) throw std::runtime_error("The archive is truncated.");
    return uint16_t(b[at] | (b[at + 1] << 8));
}

std::map<std::string, std::vector<uint8_t>> unzip(const std::vector<uint8_t>& a) {
    // end of central directory: search backwards (a comment may follow it)
    if (a.size() < 22) throw std::runtime_error("The archive is too small to be a ZIP file.");
    size_t eocd = std::string::npos;
    for (size_t i = a.size() - 22 + 1; i-- > 0 && a.size() - i < 65557;)
        if (u32(a, i) == 0x06054b50) { eocd = i; break; }
    if (eocd == std::string::npos) throw std::runtime_error("The archive has no ZIP directory.");
    const uint16_t count = u16(a, eocd + 10);
    size_t at = u32(a, eocd + 16);
    std::map<std::string, std::vector<uint8_t>> files;
    for (int i = 0; i < count; i++) {
        if (u32(a, at) != 0x02014b50) throw std::runtime_error("The ZIP directory is corrupt.");
        const uint16_t method = u16(a, at + 10);
        const uint32_t csize = u32(a, at + 20), usize = u32(a, at + 24);
        const uint16_t nameLen = u16(a, at + 28), extraLen = u16(a, at + 30), commentLen = u16(a, at + 32);
        const uint32_t local = u32(a, at + 42);
        if (at + 46 + nameLen > a.size()) throw std::runtime_error("The ZIP directory is corrupt.");
        const std::string name(a.begin() + at + 46, a.begin() + at + 46 + nameLen);
        at += 46 + nameLen + extraLen + commentLen;
        if (u32(a, local) != 0x04034b50) throw std::runtime_error("A ZIP entry is corrupt.");
        const size_t data = local + 30 + u16(a, local + 26) + u16(a, local + 28);
        if (data + csize > a.size()) throw std::runtime_error("A ZIP entry is truncated.");
        if (name.empty() || name.back() == '/') continue;
        if (method == 0) files[name] = std::vector<uint8_t>(a.begin() + data, a.begin() + data + csize);
        else if (method == 8) {
            auto out = inflateRaw(a.data() + data, csize, usize);
            if (out.size() != usize) throw std::runtime_error("A ZIP entry could not be decompressed.");
            files[name] = std::move(out);
        }
    }
    return files;
}

}  // namespace ps
