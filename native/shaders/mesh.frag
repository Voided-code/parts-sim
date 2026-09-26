#version 440
layout(location = 0) in vec3 vNormal;
layout(location = 1) in float vValue;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform Item {
    mat4 mvp;
    vec4 color;
    vec4 light;
    vec4 range;     // min, max, bands, flags (1 = colour by value, 2 = reverse, 4 = vertex colours)
};
layout(binding = 1) uniform sampler2D colormap;

void main() {
    vec3 base = color.rgb;
    int flags = int(range.w + 0.5);
    if ((flags & 1) != 0) {
        if (vValue < -1.0e37) {
            base = vec3(0.59);  // no data
        } else {
            float span = max(range.y - range.x, 1.0e-30);
            float t = clamp((vValue - range.x) / span, 0.0, 1.0);
            if (range.z > 0.5) t = (min(floor(t * range.z), range.z - 1.0) + 0.5) / range.z;
            if ((flags & 2) != 0) t = 1.0 - t;
            base = texture(colormap, vec2(t, 0.5)).rgb;
        }
    }
    if ((flags & 4) != 0) {
        // per-vertex colour (flow visuals): rgb in the normal slot, alpha in the value slot
        float a = color.a * clamp(vValue, 0.0, 1.0);
        fragColor = vec4(vNormal * a, a);
        return;
    }
    vec3 c = base;
    if (light.w > 0.5) {
        vec3 n = normalize(vNormal);
        // two-sided: face the normal towards the viewer
        if (dot(n, light.xyz) < 0.0) n = -n;
        float head = max(dot(n, normalize(light.xyz)), 0.0);
        float sky = 0.5 + 0.5 * n.y;
        float top = max(dot(n, normalize(vec3(0.2, 1.0, 0.3))), 0.0);
        c = base * (0.36 + 0.22 * sky + 0.52 * head + 0.12 * top);
    }
    fragColor = vec4(c * color.a, color.a);
}
