glslangvalidator -V shader.vert
glslangvalidator -V shader.frag
rem glslangvalidator -V light.comp -o light.spv
glslangvalidator -V forward.comp -o light.spv
rem glslangvalidator -V clustered.comp -o light.spv
rem glslangvalidator -V light_culling.comp -o light.spv
pause