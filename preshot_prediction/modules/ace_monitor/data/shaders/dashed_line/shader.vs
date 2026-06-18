#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aColor;

out vec4 Color;
uniform vec4 tint_color=vec4(1,1,1,1);

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec2 TexCoords;

void main()
{
    Color = vec4(aColor,1) * tint_color;
    vec4 pos = projection * view * model * vec4(aPos, 1.0);

    gl_Position=pos;
    TexCoords=aTexCoords;
}
