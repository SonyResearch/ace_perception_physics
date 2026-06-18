#version 330 core
out vec4 FragColor;

in vec3 Color;

uniform vec4 tint_color=vec4(1,1,1,1);

void main()
{
    FragColor = tint_color* vec4(Color,1);
}
