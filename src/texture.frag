#version 150 core
in vec2 v_Texcoords;
out vec4 fragment;
uniform sampler2D image;
void main()
{
    fragment = texture2D(image, v_Texcoords);
}
