#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cstring>
#include <array>
#include <set>
#include <unordered_map>
#include <random>

#include "VDeleter.h"
#include "SceneStructs.h"
#include "Utils.h"
#include "VulkanUtils.h"
#include "DeviceUtils.h"
#include "DefaultVkInfo.h"


//using namespace SceneStructs;

#define WIDTH 1200
#define HEIGHT 1000

const std::string MODEL_PATH = "../../res/models/sponza.obj";
const std::string TEXTURE_PATH = "../../res/textures/kamen.jpg";
const std::string NORMAP_PATH = "../../res/textures/KAMEN-bump.jpg";

const std::string VERT_SHADER = "../../shaders/vert.spv";
const std::string FRAG_SHADER = "../../shaders/frag.spv";
const std::string LIGHT_SHADER = "../../shaders/light.spv";

#ifndef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif


// Mouse control variables
enum ControlState { NONE = 0, ROTATE, TRANSLATE };
ControlState mouseState = ControlState::NONE;

glm::vec2 screenPos;
glm::vec3 trans;
glm::vec2 rotate;
float scale = 1.f;
glm::vec3 F, R, U, P;


class VulkanApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
	}
	
private:
	GLFWwindow* window;

	VDeleter<VkInstance> instance{ vkDestroyInstance };
	VDeleter<VkDebugReportCallbackEXT> callback{ instance, VkDebug::DestroyDebugReportCallbackEXT };
	VDeleter<VkSurfaceKHR> surface{ instance, vkDestroySurfaceKHR };

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VDeleter<VkDevice> device{ vkDestroyDevice };

	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkQueue computeQueue;

	// Swap chain and image retrieval
	VDeleter<VkSwapchainKHR> swapChain{ device, vkDestroySwapchainKHR };
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	// Allows us to access the images and use them as color targets
	std::vector<VDeleter<VkImageView>> swapChainImageViews;
	std::vector<VDeleter<VkFramebuffer>> swapChainFramebuffers;

	VDeleter<VkRenderPass> renderPass{ device, vkDestroyRenderPass };
	VDeleter<VkDescriptorSetLayout> descriptorSetLayout{ device, vkDestroyDescriptorSetLayout };
	VDeleter<VkPipelineLayout> pipelineLayout{ device, vkDestroyPipelineLayout };
	VDeleter<VkPipeline> graphicsPipeline{ device, vkDestroyPipeline };

	VDeleter<VkCommandPool> commandPool{ device, vkDestroyCommandPool };

	VDeleter<VkImage> depthImage{ device, vkDestroyImage };
	VDeleter<VkDeviceMemory> depthImageMemory{ device, vkFreeMemory };
	VDeleter<VkImageView> depthImageView{ device, vkDestroyImageView };

	// Mesh obj
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VDeleter<VkBuffer> vertexBuffer{ device, vkDestroyBuffer };
	VDeleter<VkDeviceMemory> vertexBufferMemory{ device, vkFreeMemory };
	VDeleter<VkBuffer> indexBuffer{ device, vkDestroyBuffer };
	VDeleter<VkDeviceMemory> indexBufferMemory{ device, vkFreeMemory };

	// UBO for MVP matrices
	VDeleter<VkBuffer> uniformStagingBuffer{ device, vkDestroyBuffer };
	VDeleter<VkDeviceMemory> uniformStagingBufferMemory{ device, vkFreeMemory };
	VDeleter<VkBuffer> uniformBuffer{ device, vkDestroyBuffer };
	VDeleter<VkDeviceMemory> uniformBufferMemory{ device, vkFreeMemory };

	// UBO for lights
	VDeleter<VkBuffer> lightStagingBuffer{ device, vkDestroyBuffer };
	VDeleter<VkDeviceMemory> lightStagingBufferMemory{ device, vkFreeMemory };
	VDeleter<VkBuffer> lightBuffer{ device, vkDestroyBuffer };
	VDeleter<VkDeviceMemory> lightBufferMemory{ device, vkFreeMemory };

	// Need to actually keep a persistent lights array on the CPU
	uboLights lightData;

	VDeleter<VkDescriptorPool> descriptorPool{ device, vkDestroyDescriptorPool };
	VkDescriptorSet descriptorSet;

	std::vector<VkCommandBuffer> commandBuffers;

	VDeleter<VkSemaphore> imageAvailableSemaphore{ device, vkDestroySemaphore };
	VDeleter<VkSemaphore> renderFinishedSemaphore{ device, vkDestroySemaphore };

	// Normal map
	VDeleter<VkImage> norMapImage{ device, vkDestroyImage };
	VDeleter<VkDeviceMemory> norMapImageMemory{ device, vkFreeMemory };
	VDeleter<VkImageView> norMapImageView{ device, vkDestroyImageView };
	VDeleter<VkSampler> norMapSampler{ device, vkDestroySampler };

	// Texture
	VDeleter<VkImage> textureImage{ device, vkDestroyImage };
	VDeleter<VkDeviceMemory> textureImageMemory{ device, vkFreeMemory };
	VDeleter<VkImageView> textureImageView{ device, vkDestroyImageView };
	VDeleter<VkSampler> textureSampler{ device, vkDestroySampler };


	// Allocations for a compute shader to move the lights around
	VDeleter<VkBuffer> lightStorageA{ device, vkDestroyBuffer };
	//VDeleter<VkBuffer> lightStorageB{ device, vkDestroyBuffer };
	VDeleter<VkDeviceMemory> lightStorageAMemory{ device, vkFreeMemory };
	//VDeleter<VkDeviceMemory> lightStorageBMemory{ device, vkFreeMemory };

	VDeleter<VkDescriptorSetLayout> lightDescriptorSetLayout{ device, vkDestroyDescriptorSetLayout };
	VDeleter<VkPipelineLayout> lightPipelineLayout{ device, vkDestroyPipelineLayout };
	VDeleter<VkPipeline> lightPipeline{ device, vkDestroyPipeline };
	VkDescriptorSet lightDescriptorSets[2];

	VDeleter<VkCommandPool> lightCommandPool{ device, vkDestroyCommandPool };
	VkCommandBuffer lightCommandBuffer = VK_NULL_HANDLE;

	VDeleter<VkFence> lightFence{ device, vkDestroyFence };

	void createComputeLightMoveBuffers(){
		// Create the lights
		std::default_random_engine rng;
		std::uniform_real_distribution<float> u01(0.f, 1.f);
		
		std::vector<Light> lights(numLights);
		for (int i = 0; i < numLights; i++) {
			glm::vec3 pos;
			pos.x = (u01(rng) - 0.5f) * 20.f;
			pos.y = (u01(rng)) * 30.f;
			pos.z = (u01(rng) - 0.5f) * 40.f;

			glm::vec4 col;
			col.r = u01(rng);
			col.g = u01(rng);
			col.b = u01(rng);
			col.a = 1.f;

			lights[i].pos = glm::vec4(pos, u01(rng) * 10.f + 1.f);
			lights[i].col = col;
			lights[i].vel = { 0.1f, 40.f, 0.0f, 0.0f };
		}

		VkDeviceSize bufferSize = lights.size() * sizeof(Light);

		VDeleter<VkBuffer> stagingBuffer{ device, vkDestroyBuffer };
		VDeleter<VkDeviceMemory> stagingBufferMemory{ device, vkFreeMemory };
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, lights.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, lightStorageA, lightStorageAMemory);

		copyBuffer(stagingBuffer, lightStorageA, bufferSize);
	}

	void createLightDescriptorSetLayout() {
		VkDescriptorSetLayoutBinding storageBufferABinding = {};
		storageBufferABinding.binding = 0;
		storageBufferABinding.descriptorCount = 1;
		storageBufferABinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		storageBufferABinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		std::array<VkDescriptorSetLayoutBinding, 1> bindings = { storageBufferABinding };
		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = bindings.size();
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, lightDescriptorSetLayout.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	void createLightComputePipeline(){

		VkDescriptorSetLayout setLayouts[] = { lightDescriptorSetLayout };
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = setLayouts;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, lightPipelineLayout.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		auto compShaderCode = readFile(LIGHT_SHADER);

		VDeleter<VkShaderModule> compShaderModule{ device, vkDestroyShaderModule };
		VkUtils::createShaderModule(device, compShaderCode, compShaderModule);
		VkPipelineShaderStageCreateInfo compShaderStageInfo = VkInfoUtils::createShaderStageInfo();
		compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		compShaderStageInfo.module = compShaderModule;

		VkComputePipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineInfo.stage = compShaderStageInfo;
		pipelineInfo.layout = lightPipelineLayout;
		pipelineInfo.pNext = nullptr;
		pipelineInfo.flags = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // not deriving from existing pipeline
		pipelineInfo.basePipelineIndex = -1; // Optional

		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, lightPipeline.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}
	}

	void createLightCommandPool() {
		PDevUtil::QueueFamilyIndices queueFamilyIndices = PDevUtil::findQueueFamilies(physicalDevice, surface);

		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily;
		if (vkCreateCommandPool(device, &poolInfo, nullptr, lightCommandPool.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics command pool!");
		}
	}

	void createLightDescriptorSet(){
		VkDescriptorSetLayout layouts[] = { lightDescriptorSetLayout, lightDescriptorSetLayout };
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 2;
		allocInfo.pSetLayouts = layouts;

		if (vkAllocateDescriptorSets(device, &allocInfo, lightDescriptorSets) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor set!");
		}

		VkDescriptorBufferInfo bufferA = {};
		bufferA.buffer = lightStorageA;
		bufferA.offset = 0;
		bufferA.range = sizeof(Light) * numLights;

		std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = lightDescriptorSets[0];
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &bufferA;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = lightDescriptorSets[1];
		descriptorWrites[1].dstBinding = 0;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &bufferA;

		vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
	}

	void createLightCommandBuffers() {

		// Wait to finish
		vkWaitForFences(device, 1, &lightFence, VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &lightFence);

		if (lightCommandBuffer) {
			vkFreeCommandBuffers(device, commandPool, 1, &lightCommandBuffer);
			lightCommandBuffer = VK_NULL_HANDLE;
		}

		// Make new command buffer
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 1;

		if (vkAllocateCommandBuffers(device, &allocInfo, &lightCommandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

		vkBeginCommandBuffer(lightCommandBuffer, &beginInfo);

		PDevUtil::QueueFamilyIndices queueFamilyIndices = PDevUtil::findQueueFamilies(physicalDevice, surface);

		VkBufferMemoryBarrier bufferBarrier = {};
		bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bufferBarrier.buffer = lightStorageA;
		bufferBarrier.size = sizeof(Light) * numLights;
		bufferBarrier.srcAccessMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		bufferBarrier.dstAccessMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarrier.dstQueueFamilyIndex = queueFamilyIndices.computeFamily;
		bufferBarrier.srcQueueFamilyIndex = queueFamilyIndices.computeFamily;

		vkCmdPipelineBarrier(lightCommandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &bufferBarrier, 0, nullptr);
		vkCmdBindPipeline(lightCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, lightPipeline);
		vkCmdBindDescriptorSets(lightCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, lightPipelineLayout, 0, 1, lightDescriptorSets, 0, 0);
		vkCmdDispatch(lightCommandBuffer, numLights / 16, 1, 1);

		bufferBarrier.srcAccessMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		bufferBarrier.dstAccessMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarrier.dstQueueFamilyIndex = queueFamilyIndices.computeFamily;
		bufferBarrier.srcQueueFamilyIndex = queueFamilyIndices.computeFamily;
		vkCmdPipelineBarrier(lightCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 1, &bufferBarrier, 0, nullptr);

		if (vkEndCommandBuffer(lightCommandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	// For vertices
	void createVertexBuffer() {
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		VDeleter<VkBuffer> stagingBuffer{ device, vkDestroyBuffer };
		VDeleter<VkDeviceMemory> stagingBufferMemory{ device, vkFreeMemory };
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
	}

	void createIndexBuffer() {
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VDeleter<VkBuffer> stagingBuffer{ device, vkDestroyBuffer };
		VDeleter<VkDeviceMemory> stagingBufferMemory{ device, vkFreeMemory };
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);
	}


	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

		glfwSetMouseButtonCallback(window, mouseButtonCallback);
		glfwSetCursorPosCallback(window, mouseMotionCallback);
		glfwSetScrollCallback(window, mouseWheelCallback);

		glfwSetWindowUserPointer(window, this);
		glfwSetWindowSizeCallback(window, VulkanApplication::onWindowResized);
	}

	void initVulkan() {
		createInstance();
		VkDebug::setupDebugCallback(instance, callback, enableValidationLayers);
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();

		// Set up graphics pipeline
		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createDepthResources();
		createFramebuffers();

		// Texture and normal map
		setUpTexture(TEXTURE_PATH, textureImage, textureImageMemory, textureImageView, textureSampler);
		setUpTexture(NORMAP_PATH, norMapImage, norMapImageMemory, norMapImageView, norMapSampler);

		// Load the obj.
		loadModel(MODEL_PATH, vertices, indices);

		// Create buffers
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffer(sizeof(UniformBufferObject), uniformStagingBuffer, uniformStagingBufferMemory, uniformBuffer, uniformBufferMemory);
		createUniformBuffer(sizeof(uboLights), lightStagingBuffer, lightStagingBufferMemory, lightBuffer, lightBufferMemory);
		
		// Lights
		createLights();

		createComputeLightMoveBuffers();
		createLightDescriptorSetLayout();

		createDescriptorPool();
		createDescriptorSet();
		createCommandBuffers();

		// Light compute shader

		createLightComputePipeline();
		//createLightCommandPool();
		createFence();
		
		createLightDescriptorSet();
		createLightCommandBuffers();

		createSemaphores();
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();

			updateUniformBuffer();
			updateLightBuffer();
			drawFrame();
		}

		vkDeviceWaitIdle(device);
	}

	/***********************************
	*       Mouse controls here        *
	************************************/
	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
	{
		if (action == GLFW_PRESS)
		{
			if (button == GLFW_MOUSE_BUTTON_LEFT)
			{
				mouseState = ControlState::ROTATE;
			}
			else if (button == GLFW_MOUSE_BUTTON_RIGHT)
			{
				mouseState = ControlState::TRANSLATE;
			}

		}
		else if (action == GLFW_RELEASE)
		{
			mouseState = ControlState::NONE;
		}
	}

	static void mouseMotionCallback(GLFWwindow* window, double xpos, double ypos)
	{
		const float s_r = 0.25f;
		const float s_t = 0.02f;

		double diffx = xpos - screenPos[0];
		double diffy = ypos - screenPos[1];
		screenPos[0] = xpos;
		screenPos[1] = ypos;

		if (mouseState == ROTATE)
		{
			//rotate
			rotate[0] += (float)s_r * (float)diffy;
			rotate[1] += (float)-s_r * (float)diffx;
		}
		else if (mouseState == TRANSLATE)
		{
			//translate
			trans[2] += (float)(-s_t * (float)diffx);
			trans[1] += (float)(s_t * (float)diffy);
		}
	}

	static void mouseWheelCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		const float s = 1.f;	// sensitivity
		trans[0] += (float)(-s * yoffset);
	}

	// End Mouse controls

	/***********************************
	*            Draw updates          *
	************************************/
	void updateUniformBuffer() {
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() / 1000.0f;

		UniformBufferObject ubo = {};
		// Model positions
		//ubo.model = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		//ubo.model = glm::inverse(glm::translate(trans)
		//	* glm::rotate(rotate[0], glm::vec3(1.0f, 0.0f, 0.0f))
		//	* glm::rotate(rotate[1], glm::vec3(0.0f, 1.0f, 0.0f)));

		F.x = cos(glm::radians(rotate[1])) * cos(glm::radians(rotate[0]));
		F.y = sin(glm::radians(rotate[0]));
		F.z = sin(glm::radians(rotate[1])) * cos(glm::radians(rotate[0]));
		F = glm::normalize(F);
		R = glm::normalize(glm::cross(F, glm::vec3(0.f, 1.f, 0.f)));
		U = glm::normalize(glm::cross(R, F));

		P = trans;

		// Camera Positions
		ubo.view = glm::lookAt(P, P + F, U);
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 100.0f);
		ubo.proj[1][1] *= -1;
		ubo.cPos = P;

		void* data;
		vkMapMemory(device, uniformStagingBufferMemory, 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, uniformStagingBufferMemory);

		copyBuffer(uniformStagingBuffer, uniformBuffer, sizeof(ubo));
	}

	// This function should do the following things:
	//  1. Acquire image from swap chain
	//  2. Execute command buffer for image
	//  3. Return image to swapchain
	void drawFrame() {
		// Retrieve image from swapchain
		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		{
			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.waitSemaphoreCount = 0;
			submitInfo.pWaitSemaphores = nullptr;
			submitInfo.pWaitDstStageMask = nullptr;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &lightCommandBuffer;
			submitInfo.signalSemaphoreCount = 0;
			submitInfo.pSignalSemaphores = nullptr;

			if (vkQueueSubmit(computeQueue, 1, &submitInfo, lightFence) != VK_SUCCESS) {
				throw std::runtime_error("failed to submit draw command buffer!");
			}

			vkWaitForFences(device, 1, &lightFence, VK_TRUE, UINT64_MAX);
			vkResetFences(device, 1, &lightFence);
		}


		// submit graphics command buffer
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		// Semaphore to wait on before starting this operation
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		// Command buffer to execute
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

		// Semaphore to update once operation finishes
		VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		// Which semaphore to wait on
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		// Swapchain and which image to present
		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		std::swap(lightDescriptorSets[1], lightDescriptorSets[0]);
	}

	// End draw updates

	/***********************************
	*           Create lights          *
	************************************/
	void createLights(){
		std::default_random_engine rng;
		std::uniform_real_distribution<float> u01(0, 1);

		for (int i = 0; i < numLights; i++) {
			glm::vec3 pos;
			pos.x = (u01(rng) - 0.5f) * 20.f;
			pos.y = (u01(rng)) * 30.f;
			pos.z = (u01(rng) - 0.5f) * 40.f;


			glm::vec3 col;
			col.r = u01(rng);
			col.g = u01(rng);
			col.b = u01(rng);


			lightData.lightPos[i] = glm::vec4(pos, u01(rng) * 10.f + 1.f);
			lightData.lightCol[i] = col;

			float distance = glm::length(pos - vertices[0].pos);
			//std::cout << distance - lightData.lightPos[i][3] << std::endl;
		}
	}

	void updateLightBuffer(){
		for (int i = 0; i < numLights; i++) {
			lightData.lightPos[i][1] += light_dt;
			if (lightData.lightPos[i][1] >= light_mx)
				lightData.lightPos[i][1] = light_mn;
		}

		// Update data on GPU
		void* data;
		vkMapMemory(device, lightStagingBufferMemory, 0, sizeof(lightData), 0, &data);
		memcpy(data, &lightData, sizeof(lightData));
		vkUnmapMemory(device, lightStagingBufferMemory);

		copyBuffer(lightStagingBuffer, lightBuffer, sizeof(lightData));
	}

	// End creating lights

	/***********************************
	*          Compute Shader          *
	************************************/


	// End Compute shader


	/***********************************
	*    Descriptor Set and Layout     *
	************************************/
	void createDescriptorSetLayout() {
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding lightLayoutBinding = {};
		lightLayoutBinding.binding = 2;
		lightLayoutBinding.descriptorCount = 1;
		lightLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		lightLayoutBinding.pImmutableSamplers = nullptr;
		lightLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding normapLayoutBinding = {};
		normapLayoutBinding.binding = 3;
		normapLayoutBinding.descriptorCount = 1;
		normapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		normapLayoutBinding.pImmutableSamplers = nullptr;
		normapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 4> bindings = { uboLayoutBinding, samplerLayoutBinding, lightLayoutBinding, normapLayoutBinding };
		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = bindings.size();
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	void createDescriptorPool() {
		std::array<VkDescriptorPoolSize, 3> poolSizes = {};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = 2;
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = 2;
		poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[2].descriptorCount = 5;

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = poolSizes.size();
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = 3;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, descriptorPool.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createDescriptorSet() {
		VkDescriptorSetLayout layouts[] = { descriptorSetLayout };
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = layouts;

		if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor set!");
		}

		VkDescriptorBufferInfo bufferInfo = {};
		bufferInfo.buffer = uniformBuffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(UniformBufferObject);

		VkDescriptorBufferInfo lightInfo = {};
		lightInfo.buffer = lightStorageA;
		lightInfo.offset = 0;
		lightInfo.range = sizeof(Light) * numLights;

		VkDescriptorImageInfo imageInfo = {};
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo.imageView = textureImageView;
		imageInfo.sampler = textureSampler;

		VkDescriptorImageInfo norMapInfo = {};
		norMapInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		norMapInfo.imageView = norMapImageView;
		norMapInfo.sampler = norMapSampler;

		std::array<VkWriteDescriptorSet, 4> descriptorWrites = {};

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = descriptorSet;
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &bufferInfo;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = descriptorSet;
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pImageInfo = &imageInfo;

		descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[2].dstSet = descriptorSet;
		descriptorWrites[2].dstBinding = 2;
		descriptorWrites[2].dstArrayElement = 0;
		descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[2].descriptorCount = 1;
		descriptorWrites[2].pBufferInfo = &lightInfo;

		descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[3].dstSet = descriptorSet;
		descriptorWrites[3].dstBinding = 3;
		descriptorWrites[3].dstArrayElement = 0;
		descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorWrites[3].descriptorCount = 1;
		descriptorWrites[3].pImageInfo = &norMapInfo;

		vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
	}

	// End descriptor set

	/***********************************
	*          Image Loading           *
	************************************/

	void setUpTexture(
		std::string tex_path,
		VDeleter<VkImage> &textureImage,
		VDeleter<VkDeviceMemory> &textureImageMemory,
		VDeleter<VkImageView> &textureImageView,
		VDeleter<VkSampler> &textureSampler) {

		createTextureImage(tex_path, textureImage, textureImageMemory);
		createTextureImageView(textureImage, textureImageView);
		createTextureSampler(textureSampler);
	}

	void createTextureImage(
		std::string tex_path,
		VDeleter<VkImage> &textureImage,
		VDeleter<VkDeviceMemory> &textureImageMemory) {
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;

		if (!pixels) {
			throw std::runtime_error("failed to load texture image!");
		}

		VDeleter<VkImage> stagingImage{ device, vkDestroyImage };
		VDeleter<VkDeviceMemory> stagingImageMemory{ device, vkFreeMemory };
		createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingImage, stagingImageMemory);

		void* data;
		vkMapMemory(device, stagingImageMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, (size_t)imageSize);
		vkUnmapMemory(device, stagingImageMemory);

		stbi_image_free(pixels);

		createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

		transitionImageLayout(stagingImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyImage(stagingImage, textureImage, texWidth, texHeight);

		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}

	void createTextureImageView(VDeleter<VkImage> &textureImage, VDeleter<VkImageView> &textureImageView) {
		createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, textureImageView);
	}

	void createTextureSampler(VDeleter<VkSampler> &textureSampler) {
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = 16;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

		if (vkCreateSampler(device, &samplerInfo, nullptr, textureSampler.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	// End image loading

	static void onWindowResized(GLFWwindow* window, int width, int height) {
		if (width == 0 || height == 0) return;

		VulkanApplication* app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(window));
		app->recreateSwapChain();
	}

	void recreateSwapChain() {
		vkDeviceWaitIdle(device);

		createSwapChain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createDepthResources();
		createFramebuffers();
		createCommandBuffers();
	}

	// Instance creation.
	void createInstance() {
		// Check that desired validation layers are available if enabled.
		if (enableValidationLayers && !VkUtils::checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}

		// This is information about the application. For the most part, this
		// information is optional (but useful to have).
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "ClusteredShader";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		// This is not optional. This information tells the Vulkan driver what 
		// extensions and validations layers we want to use. These define global
		// settings for the entire APPLICATION.
		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		// Find the extentions that GLFW needs to interface with the window
		// system. See VulkanUtils.h
		auto extensions = VkUtils::getRequiredExtensions(enableValidationLayers);
		createInfo.enabledExtensionCount = extensions.size();
		createInfo.ppEnabledExtensionNames = extensions.data();

		// Determine which global validation layers to use. Either load the
		// desired validation layers or use none.
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = VkUtils::validationLayers.size();
			createInfo.ppEnabledLayerNames = VkUtils::validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		// Create our instance
		if (vkCreateInstance(&createInfo, nullptr, instance.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
	}

	// Create an abstract window to render to. This window is backed by the real
	// GLFW window.
	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, surface.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	// Choose a GPU to use for the application
	void pickPhysicalDevice() {
		// Find number of available devices with Vulkan support.
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		// Get device handles
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		// For each possible device, we check if it is suitable for this
		// application. If so, select the first device that fits the criterions.
		for (const auto& device : devices) {
			if (PDevUtil::isDeviceSuitable(device, surface)) {
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	// Create a logical device which will be our interface with the GPU
	void createLogicalDevice() {
		// Find all the queue families that we need
		PDevUtil::QueueFamilyIndices indices = PDevUtil::findQueueFamilies(physicalDevice, surface);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<int> uniqueQueueFamilies = { indices.graphicsFamily, indices.presentFamily, indices.computeFamily };

		// Queue priority determines scheduling for multiple queues.
		float queuePriority = 1.0f;
		for (int queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		// Specify device features to use. These are queried for support for 
		// special shaders and whatnot.
		VkPhysicalDeviceFeatures deviceFeatures = {};

		// Create the actual logical device here
		VkDeviceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		// Information about queue creation
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.size();

		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = PDevUtil::deviceExtensions.size();
		createInfo.ppEnabledExtensionNames = PDevUtil::deviceExtensions.data();

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = VkUtils::validationLayers.size();
			createInfo.ppEnabledLayerNames = VkUtils::validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, device.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}
		// Retrieve the queue handles for use in the application.
		vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
		vkGetDeviceQueue(device, indices.computeFamily, 0, &computeQueue);
	}

	// Creates the swap chain
	void createSwapChain() {
		PDevUtil::SwapChainSupportDetails swapChainSupport = PDevUtil::querySwapChainSupport(physicalDevice, surface);

		VkSurfaceFormatKHR surfaceFormat = SwapchainUtil::chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = SwapchainUtil::chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = SwapchainUtil::chooseSwapExtent(swapChainSupport.capabilities, WIDTH, HEIGHT);

		// Decide number of images in the swap chain queue. This is set to the
		// min number + 1
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		// Struct to create the swap chain
		VkSwapchainCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		PDevUtil::QueueFamilyIndices indices = PDevUtil::findQueueFamilies(physicalDevice, surface);
		uint32_t queueFamilyIndices[] = { (uint32_t)indices.graphicsFamily, (uint32_t)indices.presentFamily };

		// Determine how images are shared across queue families. Exclusive 
		// allows only one family to own the image at a time and ownership must
		// be explicitly transferred.
		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		// Transformation on the image (ex. 90 deg rotation). currentTransform 
		// means do nothing.
		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;

		VkSwapchainKHR oldSwapChain = swapChain;
		createInfo.oldSwapchain = oldSwapChain;

		VkSwapchainKHR newSwapChain;
		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &newSwapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		swapChain = newSwapChain;

		// Retrieve the image handles from the swapchain
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	// Retrieves the image view handles and sets image targets
	void createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VDeleter<VkImageView>& imageView) {
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;

		viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(device, &viewInfo, nullptr, imageView.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}
	}

	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size(), VDeleter < VkImageView > {device, vkDestroyImageView});

		for (uint32_t i = 0; i < swapChainImages.size(); i++) {
			createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, swapChainImageViews[i]);
		}
	}


	// Pipeline. A lot of these settings can be factored out to a utility file.
	void createGraphicsPipeline() {
		auto vertShaderCode = readFile(VERT_SHADER);
		auto fragShaderCode = readFile(FRAG_SHADER);

		// Vulkan wrappers for shader code
		VDeleter<VkShaderModule> vertShaderModule{ device, vkDestroyShaderModule };
		VDeleter<VkShaderModule> fragShaderModule{ device, vkDestroyShaderModule };
		VkUtils::createShaderModule(device, vertShaderCode, vertShaderModule);
		VkUtils::createShaderModule(device, fragShaderCode, fragShaderModule);

		// Below we assign the shader code to a shading stage

		// Describe which stage the shader belongs to
		VkPipelineShaderStageCreateInfo vertShaderStageInfo = VkInfoUtils::createShaderStageInfo();
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = VkInfoUtils::createShaderStageInfo();
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		// Describe how vertices are sent to the shader, whether per instance
		// or per vertex
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = VkInfoUtils::createAssemblyStageInfo();

		// What part of the framebuffer to render to
		VkViewport viewport = VkInfoUtils::createViewportInfo(swapChainExtent.height, swapChainExtent.width);

		// A mask to determine how much of the image is visible
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		// Enable MSAA here? There are a bunch of other things that need to be 
		// tweaked I think to make this work.
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		VkDescriptorSetLayout setLayouts[] = { descriptorSetLayout };
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = setLayouts;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, pipelineLayout.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, graphicsPipeline.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}
	}

	// Renderpass specifies how many color and depth buffers there will be
	void createRenderPass() {
		VkAttachmentDescription colorAttachment = {};
		colorAttachment.format = swapChainImageFormat;						// Color format should be same as swap chain
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;					// For multisampling
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentDescription depthAttachment = {};
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef = {};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subPass = {};
		subPass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subPass.colorAttachmentCount = 1;
		subPass.pColorAttachments = &colorAttachmentRef;
		subPass.pDepthStencilAttachment = &depthAttachmentRef;

		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = attachments.size();
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subPass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, renderPass.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	// Holds attachments for each image
	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size(), VDeleter < VkFramebuffer > {device, vkDestroyFramebuffer});

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::array<VkImageView, 2> attachments = {
				swapChainImageViews[i],
				depthImageView
			};

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = attachments.size();
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, swapChainFramebuffers[i].replace()) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void createCommandPool() {
		PDevUtil::QueueFamilyIndices queueFamilyIndices = PDevUtil::findQueueFamilies(physicalDevice, surface);

		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily;

		if (vkCreateCommandPool(device, &poolInfo, nullptr, commandPool.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics command pool!");
		}
	}

	void createDepthResources() {
		VkFormat depthFormat = findDepthFormat();

		createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
		createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, depthImageView);

		transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}

	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}

	VkFormat findDepthFormat() {
		return findSupportedFormat(
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	/****  Uniform Buffers ****/
	void createUniformBuffer(
		int size,
		VDeleter<VkBuffer> &uniformStagingBuffer,
		VDeleter<VkDeviceMemory> &uniformStagingBufferMemory,
		VDeleter<VkBuffer> &uniformBuffer,
		VDeleter<VkDeviceMemory> &uniformBufferMemory) {
		VkDeviceSize bufferSize = size;

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformStagingBuffer, uniformStagingBufferMemory);
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, uniformBuffer, uniformBufferMemory);
	}


	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VDeleter<VkImage>& image, VDeleter<VkDeviceMemory>& imageMemory) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, image.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, imageMemory.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;

		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (hasStencilComponent(format)) {
				barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			}
		}
		else {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}

		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		if (oldLayout == VK_IMAGE_LAYOUT_PREINITIALIZED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_PREINITIALIZED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
			);

		endSingleTimeCommands(commandBuffer);
	}

	void copyImage(VkImage srcImage, VkImage dstImage, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageSubresourceLayers subResource = {};
		subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subResource.baseArrayLayer = 0;
		subResource.mipLevel = 0;
		subResource.layerCount = 1;

		VkImageCopy region = {};
		region.srcSubresource = subResource;
		region.dstSubresource = subResource;
		region.srcOffset = { 0, 0, 0 };
		region.dstOffset = { 0, 0, 0 };
		region.extent.width = width;
		region.extent.height = height;
		region.extent.depth = 1;

		vkCmdCopyImage(
			commandBuffer,
			srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &region
			);

		endSingleTimeCommands(commandBuffer);
	}


	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VDeleter<VkBuffer>& buffer, VDeleter<VkDeviceMemory>& bufferMemory) {
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, buffer.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, bufferMemory.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion = {};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}


	void createCommandBuffers() {
		if (commandBuffers.size() > 0) {
			vkFreeCommandBuffers(device, commandPool, commandBuffers.size(), commandBuffers.data());
		}

		commandBuffers.resize(swapChainFramebuffers.size());

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}

		for (size_t i = 0; i < commandBuffers.size(); i++) {
			// Describes how the commands should be used. Here they are going
			// to be simultaneously resubmitted while executing.
			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

			vkBeginCommandBuffer(commandBuffers[i], &beginInfo);

			// Information for starting the render pass.
			VkRenderPassBeginInfo renderPassInfo = {};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[i];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = swapChainExtent;

			std::array<VkClearValue, 2> clearValues = {};
			clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
			clearValues[1].depthStencil = { 1.0f, 0 };

			renderPassInfo.clearValueCount = clearValues.size();
			renderPassInfo.pClearValues = clearValues.data();

			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

			VkBuffer vertexBuffers[] = { vertexBuffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

			vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

			vkCmdDrawIndexed(commandBuffers[i], indices.size(), 1, 0, 0, 0);

			vkCmdEndRenderPass(commandBuffers[i]);

			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}

	void createSemaphores() {
		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, imageAvailableSemaphore.replace()) != VK_SUCCESS ||
			vkCreateSemaphore(device, &semaphoreInfo, nullptr, renderFinishedSemaphore.replace()) != VK_SUCCESS) {

			throw std::runtime_error("failed to create semaphores!");
		}
	}

	void createFence() {
		VkFenceCreateInfo fenceInfo = {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.pNext = nullptr;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		if (vkCreateFence(device, &fenceInfo, nullptr, lightFence.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create fence!");
		}

	}

};