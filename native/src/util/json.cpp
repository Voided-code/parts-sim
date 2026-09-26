#include "json.hpp"

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace ps::json {

static const Value NIL{};

const Value& Value::operator[](const std::string& key) const {
    if (type != Object) return NIL;
    auto it = obj.find(key);
    return it == obj.end() ? NIL : it->second;
}

const Value& Value::operator[](size_t i) const { return type == Array && i < arr.size() ? arr[i] : NIL; }

namespace {

struct Parser {
    const std::string& s;
    size_t i = 0;
    int depth = 0;

    [[noreturn]] void fail(const char* what) { throw std::runtime_error(std::string("Invalid JSON: ") + what); }
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) i++; }
    bool lit(const char* w) {
        const size_t n = std::char_traits<char>::length(w);
        if (s.compare(i, n, w) == 0) { i += n; return true; }
        return false;
    }

    Value value() {
        if (++depth > 256) fail("nested too deeply");
        ws();
        if (i >= s.size()) fail("unexpected end");
        Value v;
        const char c = s[i];
        if (c == '{') {
            v.type = Value::Object;
            i++;
            ws();
            if (i < s.size() && s[i] == '}') { i++; depth--; return v; }
            for (;;) {
                ws();
                if (i >= s.size() || s[i] != '"') fail("expected a key");
                std::string key = string();
                ws();
                if (i >= s.size() || s[i] != ':') fail("expected ':'");
                i++;
                v.obj[key] = value();
                ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == '}') { i++; break; }
                fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            v.type = Value::Array;
            i++;
            ws();
            if (i < s.size() && s[i] == ']') { i++; depth--; return v; }
            for (;;) {
                v.arr.push_back(value());
                ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == ']') { i++; break; }
                fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            v.type = Value::String;
            v.str = string();
        } else if (lit("true")) { v.type = Value::Bool; v.b = true; }
        else if (lit("false")) { v.type = Value::Bool; }
        else if (lit("null")) {}
        else {
            char* end = nullptr;
            v.num = std::strtod(s.c_str() + i, &end);
            if (end == s.c_str() + i) fail("unexpected character");
            v.type = Value::Number;
            i = end - s.c_str();
        }
        depth--;
        return v;
    }

    std::string string() {
        std::string out;
        i++;  // opening quote
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                const char e = s[++i];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        if (i + 4 >= s.size()) fail("bad escape");
                        unsigned cp = unsigned(std::strtoul(s.substr(i + 1, 4).c_str(), nullptr, 16));
                        i += 4;
                        if (cp < 0x80) out += char(cp);
                        else if (cp < 0x800) { out += char(0xc0 | (cp >> 6)); out += char(0x80 | (cp & 0x3f)); }
                        else { out += char(0xe0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3f)); out += char(0x80 | (cp & 0x3f)); }
                        break;
                    }
                    default: out += e;
                }
                i++;
            } else out += s[i++];
        }
        if (i >= s.size()) fail("unterminated string");
        i++;
        return out;
    }
};

void write(std::ostringstream& o, const Value& v, int indent, int level) {
    const std::string pad = indent ? "\n" + std::string(size_t(indent * (level + 1)), ' ') : "";
    const std::string end = indent ? "\n" + std::string(size_t(indent * level), ' ') : "";
    switch (v.type) {
        case Value::Null: o << "null"; break;
        case Value::Bool: o << (v.b ? "true" : "false"); break;
        case Value::Number:
            if (std::isfinite(v.num)) o << v.num; else o << "null";
            break;
        case Value::String: {
            o << '"';
            for (char c : v.str) {
                if (c == '"' || c == '\\') o << '\\' << c;
                else if (c == '\n') o << "\\n";
                else if ((unsigned char)c < 0x20) o << "\\u00" << "0123456789abcdef"[(c >> 4) & 15] << "0123456789abcdef"[c & 15];
                else o << c;
            }
            o << '"';
            break;
        }
        case Value::Array: {
            o << '[';
            for (size_t k = 0; k < v.arr.size(); k++) { o << (k ? "," : "") << pad; write(o, v.arr[k], indent, level + 1); }
            o << (v.arr.empty() ? "" : end) << ']';
            break;
        }
        case Value::Object: {
            o << '{';
            size_t k = 0;
            for (const auto& [key, val] : v.obj) {
                o << (k++ ? "," : "") << pad;
                Value ks;
                ks.type = Value::String;
                ks.str = key;
                write(o, ks, indent, level + 1);
                o << (indent ? ": " : ":");
                write(o, val, indent, level + 1);
            }
            o << (v.obj.empty() ? "" : end) << '}';
            break;
        }
    }
}

}  // namespace

Value parse(const std::string& text) {
    Parser p{text};
    Value v = p.value();
    p.ws();
    if (p.i != text.size()) p.fail("trailing characters");
    return v;
}

std::string stringify(const Value& v, int indent) {
    std::ostringstream o;
    o.precision(17);
    write(o, v, indent, 0);
    return o.str();
}

}  // namespace ps::json
