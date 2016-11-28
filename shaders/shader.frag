#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPosition;
layout(location = 3) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;

int numLights = 40;
layout(binding = 2) uniform uboLights {
	vec4 lightPos[40];
	vec3 lightCol[40];
} lights;

void main() {
    
	vec4 color;
    for (int i = 0; i < numLights; i++) {
    	vec3 lightDir = vec3(lights.lightPos[i]) - fragPosition;
    	float distance = length(lightDir);
    	lightDir = lightDir / distance;

    	float attenuation = max(0.001f, lights.lightPos[i][3] - distance);
    	color += texture(texSampler, fragTexCoord) * attenuation * vec4(lights.lightCol[i], 1.f) * clamp(dot(fragNormal, lightDir), 0.001, 1.0); 
    }
    outColor = clamp(1.f * color, 0.001, 1.0);// + 0.2 * texture(texSampler, fragTexCoord);
    //utColor = vec4(fragNormal, 1.f);
}