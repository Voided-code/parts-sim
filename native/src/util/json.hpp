// Minimal JSON reader (for glTF headers and settings files).
#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ps::json {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    const Value& operator[](const std::string& key) const;
    const Value& operator[](size_t i) const;
    bool has(const std::string& key) const { return type == Object && obj.count(key); }
    size_t size() const { return type == Array ? arr.size() : type == Object ? obj.size() : 0; }
    double number(double fallback = 0) const { return type == Number ? num : fallback; }
    int integer(int fallback = 0) const { return type == Number ? int(num) : fallback; }
    const std::string& string() const { return str; }
};

/** Throws std::runtime_error on malformed input. */
Value parse(const std::string& text);
/** Serialize (for settings files). */
std::string stringify(const Value& v, int indent = 0);

}  // namespace ps::json
