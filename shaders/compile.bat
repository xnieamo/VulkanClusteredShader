glslangvalidator -V shader.vert
glslangvalidator -V shader.frag
rem glslangvalidator -V light.comp -o light.spv
rem glslangvalidator -V forward.comp -o light.spv
glslangvalidator -V clustered.comp -o light.spv
pause