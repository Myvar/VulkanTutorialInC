#version 450

layout(binding = 0) uniform UniformBufferObject {
    layout(column_major) mat4 model;
    layout(column_major) mat4 view;
    layout(column_major) mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    vec4 pos = vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * ubo.model * pos;
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}
