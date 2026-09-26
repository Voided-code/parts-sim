// Minimal XML element tree (tags without namespace prefix, attributes, children) for the small,
// machine-written XML inside SolidWorks and 3MF files. Not a validating parser.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ps::xml {

struct Node {
    std::string tag;
    std::map<std::string, std::string> attrs;
    std::vector<std::unique_ptr<Node>> children;

    std::string attr(const std::string& name, const std::string& fallback = "") const {
        auto it = attrs.find(name);
        return it == attrs.end() ? fallback : it->second;
    }
    std::vector<const Node*> childrenNamed(const std::string& t) const {
        std::vector<const Node*> out;
        for (const auto& c : children)
            if (c->tag == t) out.push_back(c.get());
        return out;
    }
};

std::string decodeEntities(const std::string& s);
/** Parse into a synthetic "#root" node whose children are the document's top-level elements. */
std::unique_ptr<Node> parse(const std::string& text);

}  // namespace ps::xml
