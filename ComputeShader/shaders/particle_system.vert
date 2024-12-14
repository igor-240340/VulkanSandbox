#version 450

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec3 frag_color;

void main() {
    gl_PointSize = 14.0;
    gl_Position = vec4(in_position.xy, 1.0, 1.0);
    frag_color = in_color.rgb;
}