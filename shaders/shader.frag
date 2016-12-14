#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPosition;
layout(location = 3) in vec3 fragNormal;
layout(location = 4) in vec3 camPos;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D norSampler;


struct Light 
{
	vec4 pos;
	vec4 col;
	vec4 vel;
};

layout(std430, binding = 3) buffer LightsA
{
	Light lights[];
};

layout(std430, binding = 4) buffer LookupIndex
{
	int data[];
} lookupIndexBuffer;

layout(std430, binding = 5) buffer ClustersData
{
	// ivec2 lookupIndices[];
	int lookupIndices[][2];
};

layout(origin_upper_left) in vec4 gl_FragCoord;

int numLights = 100;
const float Z_explicit[10] = {0.05f, 0.23f, 0.52f, 1.2f, 2.7f, 6.0f, 14.f, 31.f, 71.f, 161.f};

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

int findZ (float z) {

    float minDiff = 1000000.f;
    int minIdx = -1;
    for (int i = 0; i < 10; i++) {
    	float tempDiff = z - Z_explicit[i];
    	if (tempDiff < minDiff && tempDiff >= 0.f) {
    		minDiff = tempDiff;
    		minIdx = i;
    	}
    }
	return minIdx;
}

void main() {
    
	vec4 color;
	vec3 nor = texture(norSampler, fragTexCoord).xyz;
	nor = applyNormalMap(fragNormal, nor);

	vec3 viewDir = normalize(camPos - fragPosition);
	

	// Find screen space tile
    int Sx = int(gl_FragCoord.x / 32);
    int Sy = int(gl_FragCoord.y / 32);

    int X = int(WIDTH / 32) + 1;
    int Y = int(HEIGHT / 32) + 1;

    // Find Z cluster
    // http://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
    float zNear = 0.1f;
    float zFar  = 100.f;
    float z_b = gl_FragCoord.z;
    float z_n = 2.0 * z_b - 1.0;
    float z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));

    // Find Z cluster index
    int Sz = findZ(z_e);

    // Find unrolled index
    int l_idx = Sx + Sy * X;// + Sz * X * Y;

    ivec2 lightIdx = ivec2(lookupIndices[l_idx][0], lookupIndices[l_idx][1]);

   	int test = lookupIndexBuffer.data[0];
   	
   	// Debug map
   	if (false)
   		outColor = vec4(0.f, 0.f, 0.f, 1.f);
   	else 
	{

		// uint start = lightIdx[0];
		// uint end =  lightIdx[0] + lightIdx[1];
		uint start = l_idx * 10;
		uint end = l_idx * 10 + 50;
		// uint start = 0;
		// uint end = 50;

		for (uint i = start; i < end; i++) {
	    	// Light light = lights[lightIndexLookup[i]];
	    	vec4 pos = lights[lookupIndexBuffer.data[i]].pos;
	    	vec4 col = lights[lookupIndexBuffer.data[i]].col;
	    	// vec4 pos = lights[i].pos;
	    	// vec4 col = lights[i].col;
	    	vec3 lightDir = pos.xyz - fragPosition;
	    	float distance = length(lightDir);
	    	lightDir = lightDir / distance;

	    	float attenuation = max(0.001f, pos.w - distance);

	    	float specular = specularLighting(nor, lightDir, viewDir);
	    	float diffuse  = clamp(dot(nor, lightDir), 0.001, 1.0);

	    	color += texture(texSampler, fragTexCoord) * attenuation * vec4(normalize(col.rgb), 1.f) * (specular + diffuse); 
	    }
    	outColor = clamp(1.f * color, 0.001, 1.0);

	}


 //    // Real 
 //    for (int i = 0; i < numLights; i++) {

 //    	vec4 pos = lights[i].pos;
 //    	vec4 col = lights[i].col;
 //    	vec3 lightDir = pos.xyz - fragPosition;
 //    	float distance = length(lightDir);
 //    	lightDir = lightDir / distance;

 //    	float attenuation = max(0.001f, pos.w - distance);

 //    	float specular = specularLighting(nor, lightDir, viewDir);
 //    	float diffuse  = clamp(dot(nor, lightDir), 0.001, 1.0);

 //    	color += texture(texSampler, fragTexCoord) * attenuation * vec4(normalize(col.rgb), 1.f) * (specular + diffuse); 
 //    }
	// // Actual output color
 //    outColor = clamp(1.f * color, 0.001, 1.0);



    // Cluster debug
    // int zz = Sx + Sy * X + Sz * X * Y;
	// outColor = vec4((vec3(1.f) * (zz % 10))/ 10.0, 1.f);

	// Cluster debug pretty
    // outColor = vec4(normalize(vec3(Sx + 1, Sy + 1, Sz)), 1.f);

    // Depth Debug
    // outColor = vec4(vec3(1.f,1.f,1.f) * z_e / 100.f, 1.f);

    // Normal Debug
    // outColor = vec4(nor, 1.f);



}