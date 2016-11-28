#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPosition;
layout(location = 3) in vec3 fragNormal;
layout(location = 4) in vec3 camPos;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 3) uniform sampler2D norSampler;

int numLights = 100;
layout(binding = 2) uniform uboLights {
	vec4 lightPos[100];
	vec3 lightCol[100];
} lights;

vec3 applyNormalMap(vec3 geomnor, vec3 normap) {
    normap = normap * 2.0 - 1.0;
    vec3 up = normalize(vec3(0.001, 1, 0.001));
    vec3 surftan = normalize(cross(geomnor, up));
    vec3 surfbinor = cross(geomnor, surftan);
    return normap.y * surftan + normap.x * surfbinor + normap.z * geomnor;
}

float specularLighting(vec3 normal, vec3 lightDir, vec3 viewDir) {
    vec3 H = normalize(lightDir + viewDir);
    float specAngle = clamp(dot(normal, H), 0.0, 1.0);
    return pow(specAngle, 0.01) * 0.01;
}

void main() {
    
	vec4 color;
	vec3 nor = texture(norSampler, fragTexCoord).xyz;
	nor = applyNormalMap(fragNormal, nor);

	vec3 viewDir = normalize(camPos - fragPosition);

    for (int i = 0; i < numLights; i++) {
    	vec3 lightDir = vec3(lights.lightPos[i]) - fragPosition;
    	float distance = length(lightDir);
    	lightDir = lightDir / distance;

    	float attenuation = max(0.001f, lights.lightPos[i][3] - distance);

    	float specular = specularLighting(nor, lightDir, viewDir);
    	float diffuse  = clamp(dot(nor, lightDir), 0.001, 1.0);

    	color += texture(texSampler, fragTexCoord) * attenuation * vec4(lights.lightCol[i], 1.f) * (specular + diffuse); 
    }
    outColor = clamp(1.f * color, 0.001, 1.0);// + 0.2 * texture(texSampler, fragTexCoord);
    //outColor = vec4(nor, 1.f);
}