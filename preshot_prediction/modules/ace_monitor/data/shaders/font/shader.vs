#version 330 core
layout (location = 0) in vec4 aVertex;
out vec2 TexCoords;

uniform mat4 projection;

void main()
{
    gl_Position =  vec4(aVertex.xy, 0.0, 1.0);
    TexCoords = aVertex.zw;
}
