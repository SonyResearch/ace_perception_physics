#version 330 core
out vec4 FragColor;

in vec4 Color;
uniform vec4 tint_color=vec4(1,1,1,1);

in vec2 TexCoords;

uniform vec2  view_resolution;
uniform float dash_size;
uniform float gap_size;

void main()
{
    float dist=length(view_resolution*TexCoords.x);
    if (fract(dist / (dash_size + gap_size)) > dash_size/(dash_size + gap_size))
        discard;
    FragColor = Color*tint_color;
}
