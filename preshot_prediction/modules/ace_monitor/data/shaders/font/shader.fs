#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D texture_diffuse;
uniform vec4 tint_color=vec4(1,1,1,1);

void main()
{
    FragColor = vec4(tint_color.r,tint_color.g,tint_color.b,tint_color.a*texture(texture_diffuse, TexCoords).r);
}
