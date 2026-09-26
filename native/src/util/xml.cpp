#include "xml.hpp"

#include <cctype>
#include <cstdlib>

namespace ps::xml {

static void appendUtf8(std::string& out, unsigned cp) {
    if (cp < 0x80) out += char(cp);
    else if (cp < 0x800) { out += char(0xc0 | (cp >> 6)); out += char(0x80 | (cp & 0x3f)); }
    else if (cp < 0x10000) { out += char(0xe0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3f)); out += char(0x80 | (cp & 0x3f)); }
    else { out += char(0xf0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3f)); out += char(0x80 | ((cp >> 6) & 0x3f)); out += char(0x80 | (cp & 0x3f)); }
}

std::string decodeEntities(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] != '&') { out += s[i]; continue; }
        const size_t semi = s.find(';', i);
        if (semi == std::string::npos || semi - i > 10) { out += s[i]; continue; }
        const std::string e = s.substr(i + 1, semi - i - 1);
        if (e == "lt") out += '<';
        else if (e == "gt") out += '>';
        else if (e == "amp") out += '&';
        else if (e == "quot") out += '"';
        else if (e == "apos") out += '\'';
        else if (!e.empty() && e[0] == '#') {
            const bool hex = e.size() > 1 && (e[1] == 'x' || e[1] == 'X');
            appendUtf8(out, unsigned(std::strtoul(e.c_str() + (hex ? 2 : 1), nullptr, hex ? 16 : 10)));
        } else { out += s.substr(i, semi - i + 1); }
        i = semi;
    }
    return out;
}

static std::string localName(const std::string& n) {
    const size_t c = n.find(':');
    return c == std::string::npos ? n : n.substr(c + 1);
}

std::unique_ptr<Node> parse(const std::string& t) {
    auto root = std::make_unique<Node>();
    root->tag = "#root";
    std::vector<Node*> stack{root.get()};
    size_t i = 0;
    const size_t n = t.size();
    while (i < n) {
        const size_t lt = t.find('<', i);
        if (lt == std::string::npos) break;
        i = lt;
        if (t.compare(i, 4, "<!--") == 0) { const size_t e = t.find("-->", i); i = e == std::string::npos ? n : e + 3; continue; }
        if (t.compare(i, 2, "<?") == 0) { const size_t e = t.find("?>", i); i = e == std::string::npos ? n : e + 2; continue; }
        if (t.compare(i, 9, "<![CDATA[") == 0) { const size_t e = t.find("]]>", i); i = e == std::string::npos ? n : e + 3; continue; }
        if (t.compare(i, 2, "<!") == 0) { const size_t e = t.find('>', i); i = e == std::string::npos ? n : e + 1; continue; }
        if (t.compare(i, 2, "</") == 0) {
            const size_t e = t.find('>', i);
            if (stack.size() > 1) stack.pop_back();
            i = e == std::string::npos ? n : e + 1;
            continue;
        }
        // start tag
        size_t p = i + 1;
        while (p < n && !std::isspace((unsigned char)t[p]) && t[p] != '>' && t[p] != '/') p++;
        auto node = std::make_unique<Node>();
        node->tag = localName(t.substr(i + 1, p - i - 1));
        bool selfClose = false;
        while (p < n) {
            while (p < n && std::isspace((unsigned char)t[p])) p++;
            if (p >= n) break;
            if (t[p] == '>') { p++; break; }
            if (t[p] == '/') { selfClose = true; p++; continue; }
            size_t q = p;
            while (q < n && t[q] != '=' && !std::isspace((unsigned char)t[q]) && t[q] != '>' && t[q] != '/') q++;
            const std::string name = localName(t.substr(p, q - p));
            while (q < n && std::isspace((unsigned char)t[q])) q++;
            if (q < n && t[q] == '=') {
                q++;
                while (q < n && std::isspace((unsigned char)t[q])) q++;
                const char quote = q < n ? t[q] : '"';
                if (quote == '"' || quote == '\'') {
                    const size_t e = t.find(quote, q + 1);
                    node->attrs[name] = decodeEntities(t.substr(q + 1, (e == std::string::npos ? n : e) - q - 1));
                    p = e == std::string::npos ? n : e + 1;
                    continue;
                }
            }
            p = q > p ? q : p + 1;
        }
        Node* raw = node.get();
        stack.back()->children.push_back(std::move(node));
        if (!selfClose) stack.push_back(raw);
        i = p;
    }
    return root;
}

}  // namespace ps::xml
