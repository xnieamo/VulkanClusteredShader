#include "DefaultVkInfo.h"

VkPipelineShaderStageCreateInfo VkInfoUtils::createShaderStageInfo(){
	VkPipelineShaderStageCreateInfo shaderStageInfo = {};

	shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStageInfo.pName = "main";

	return shaderStageInfo;
}

VkPipelineInputAssemblyStateCreateInfo VkInfoUtils::createAssemblyStageInfo(){
	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	return inputAssembly;
}

VkViewport VkInfoUtils::createViewportInfo(int height, int width){
	VkViewport viewport = {};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)width;
	viewport.height = (float)height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	return viewport;
}
