#version 440
// Every viewport item: part surface, edges, overlays, markers. World-space vertices.
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in float value;

layout(std140, binding = 0) uniform Item {
    mat4 mvp;
    vec4 color;     // rgb, opacity
    vec4 light;     // xyz: direction towards the viewer (world), w: 1 = lit
    vec4 range;     // min, max, bands, flags (1 = colour by value, 2 = reverse)
};

layout(location = 0) out vec3 vNormal;
layout(location = 1) out float vValue;

void main() {
    vNormal = normal;
    vValue = value;
    gl_Position = mvp * vec4(position, 1.0);
}
