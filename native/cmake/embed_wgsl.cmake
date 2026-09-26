# Writes a C++ file holding every WGSL shader as a string: wgslSource("fea") etc.
# Invoked with -DOUT=<file> -DDIR=<shader folder>
file(GLOB SHADERS ${DIR}/*.wgsl)
list(SORT SHADERS)
set(body "#include <cstring>\n#include <string>\nnamespace ps {\nconst char* wgslSource(const std::string& name) {\n")
foreach(f ${SHADERS})
  get_filename_component(stem ${f} NAME_WE)
  file(READ ${f} text)
  string(APPEND body "    if (name == \"${stem}\") return R\"WGSL(${text})WGSL\";\n")
endforeach()
string(APPEND body "    return nullptr;\n}\n}  // namespace ps\n")
file(WRITE ${OUT} "${body}")
