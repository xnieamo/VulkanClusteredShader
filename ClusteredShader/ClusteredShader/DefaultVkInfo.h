#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace VkInfoUtils {
	VkPipelineShaderStageCreateInfo createShaderStageInfo();
	VkPipelineInputAssemblyStateCreateInfo createAssemblyStageInfo();
	VkViewport createViewportInfo(int height, int width);
}