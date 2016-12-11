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


struct Light 
{
	vec4 pos;
	vec4 col;
	vec4 vel;
};

layout(std140, binding = 2) buffer LightsA
{
	Light lights[];
};

int numLights = 100;
const float Z_explicit[10] = {0.1f, 0.23f, 0.52f, 1.2f, 2.7f, 6.0f, 14.f, 31.f, 71.f, 161.f};

#define WIDTH 1200
#define HEIGHT 1000

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
    	vec4 pos = lights[i].pos;
    	vec3 lightDir = pos.xyz - fragPosition;
    	float distance = length(lightDir);
    	lightDir = lightDir / distance;

    	float attenuation = max(0.001f, pos.w - distance);

    	float specular = specularLighting(nor, lightDir, viewDir);
    	float diffuse  = clamp(dot(nor, lightDir), 0.001, 1.0);

    	color += texture(texSampler, fragTexCoord) * attenuation * vec4(normalize(lights[i].col.rgb), 1.f) * (specular + diffuse); 
    }

    // Find screen space tile
    int Sx = int(gl_FragCoord.x / 32);
    int Sy = int(gl_FragCoord.y / 32);

    float X = float(WIDTH / 32);
    float Y = float(HEIGHT / 32);

    // Find Z cluster
    // http://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
    float zNear = 0.1f;
    float zFar  = 100.f;
    float z_b = gl_FragCoord.z;
    float z_n = 2.0 * z_b - 1.0;
    float z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));

    float minDiff = 1000000.f;
    int minIdx = -1;
    for (int i = 0; i < 10; i++) {
    	float tempDiff = abs(Z_explicit[i] - z_e);
    	if (tempDiff < minDiff) {
    		minDiff = tempDiff;
    		minIdx = i;
    	}
    }


    // Cluster debug
    outColor = vec4(normalize(vec3(Sx/X, Sy/Y, minIdx/10.f)), 1.f);

    // Depth Debug
    // outColor = vec4(vec3(1.f,1.f,1.f) * z_e / 100.f, 1.f);

    // Normal Debug
    // outColor = vec4(nor, 1.f);

    // Actual output color
    // outColor = clamp(1.f * color, 0.001, 1.0);// + 0.2 * texture(texSampler, fragTexCoord);

}