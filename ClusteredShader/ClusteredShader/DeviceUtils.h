#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include <set>

#include "VDeleter.h"


// Functions and structs related to loading physical devices.
namespace PDevUtil {

	// Information about the swap chains
	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;		// Min/max number of images, width, and height
		std::vector<VkSurfaceFormatKHR> formats;	// Pixel format, color space
		std::vector<VkPresentModeKHR> presentModes;	// Available presentation modes
	};

	// Need extension for swap chains since it is not Vulkan core
	const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	// Defines necessary queue families for this application
	struct QueueFamilyIndices {
		int graphicsFamily = -1;
		int presentFamily = -1;
		int computeFamily = -1;

		bool isComplete() {
			return graphicsFamily >= 0
				&& presentFamily >= 0
				&& computeFamily >= 0;
		}
	};

	// Finds all queue families supported by the device
	QueueFamilyIndices findQueueFamilies(
		const VkPhysicalDevice &device,
		VDeleter<VkSurfaceKHR> &surface) {
		QueueFamilyIndices indices;

		// Get number of queue families and load handles into vector
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		// Loop over queue families and find ones that support our needs
		int i = 0;
		for (const auto& queueFamily : queueFamilies) {

			// Check for graphics queue
			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}

			// Find compute queue
			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
				indices.computeFamily = i;
			}

			// Check for window presentation support
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
			if (queueFamily.queueCount > 0 && presentSupport) {
				indices.presentFamily = i;
			}

			if (indices.isComplete()) {
				break;
			}

			i++;
		}

		return indices;
	}

	// Checks to see if device supports all necessary extensions
	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	// Find information about swap chain
	SwapChainSupportDetails querySwapChainSupport(
		const VkPhysicalDevice &device,
		VDeleter<VkSurfaceKHR> &surface) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}


	// Check if p device is suitable for this application
	bool isDeviceSuitable(
		const VkPhysicalDevice &device,
		VDeleter<VkSurfaceKHR> &surface) {
		PDevUtil::QueueFamilyIndices indices = PDevUtil::findQueueFamilies(device, surface);

		bool extensionsSupported = PDevUtil::checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extensionsSupported) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}


}

// Choosing swapchain format
namespace SwapchainUtil {
	
	// Pick surface format which decides color space for input and display 
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		// If no preferred format, return RGB + SRGB
		if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
			return{ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
		}

		// Otherwise, loop over possible formats to find nearest match.
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	// Queue format of the swap chain. There are four possible modes:
	//   1. VK_PRESENT_MODE_IMMEDIATE_KHR    : immediate presentation
	//   2. VK_PRESENT_MODE_FIFO_KHR         : uses a queue to insert images (GUARANTEED ON EVERY DEVICE)
	//   3. VK_PRESENT_MODE_FIFO_RELAXED_KHR : uses a queue, but if last presentation was blank,
	//										   immediately transfers
	//   4. VK_PRESENT_MODE_MAILBOX_KHR      : uses queue, but images not presented are replaced with new ones 
	//                                         when the queue is full
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	// Resolution and size of the swap images
	VkExtent2D chooseSwapExtent(
		const VkSurfaceCapabilitiesKHR& capabilities,
		int width,
		int height) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		else {
			VkExtent2D actualExtent = { width, height };

			actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

			return actualExtent;
		}
	}


}