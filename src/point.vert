#version 150 core

in vec4 position;
in vec2 texcoords;
out vec2 v_Texcoords;
void main()
{
    gl_Position = position;
	v_Texcoords = texcoords;
} 