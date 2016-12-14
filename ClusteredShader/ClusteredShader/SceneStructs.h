#pragma once

#include <vulkan\vulkan.h>
#include <array>
#include <glm\glm.hpp>

namespace SceneStructs {

	struct Vertex {
		glm::vec3 pos;
		glm::vec3 color;
		glm::vec2 texCoord;
		glm::vec3 normal;

		static VkVertexInputBindingDescription getBindingDescription() {
			VkVertexInputBindingDescription bindingDescription = {};
			bindingDescription.binding = 0;
			bindingDescription.stride = sizeof(Vertex);
			bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			return bindingDescription;
		}

		static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
			std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions = {};

			attributeDescriptions[0].binding = 0;
			attributeDescriptions[0].location = 0;
			attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[0].offset = offsetof(Vertex, pos);

			attributeDescriptions[1].binding = 0;
			attributeDescriptions[1].location = 1;
			attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[1].offset = offsetof(Vertex, color);

			attributeDescriptions[2].binding = 0;
			attributeDescriptions[2].location = 2;
			attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
			attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

			attributeDescriptions[3].binding = 0;
			attributeDescriptions[3].location = 3;
			attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[3].offset = offsetof(Vertex, normal);


			return attributeDescriptions;
		}

		bool operator==(const Vertex& other) const {
			return pos == other.pos && color == other.color && texCoord == other.texCoord && normal == other.normal;
		}
	};

	struct UniformBufferObject {
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
		glm::vec3 cPos;
	};

	struct ComputeUBO {
		int height;
		int width;
		int xtiles;
		int ytiles;
		int max_lights_per_cluster;
		int number_lights;
		int tile_size, c;
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
		glm::vec4 pos;
		glm::vec4 dir;
		glm::vec4 up;

	};

	// Lights
	const int numLights = 200;
	struct uboLights {
		// Using the fourth entry in lightPos as the radius
		glm::vec4 lightPos[numLights];
		glm::vec3 lightCol[numLights];
	};

	const float light_dt = 0.02f;
	const float light_mx = 30.f;
	const float light_mn = 0.f;

	struct Light {
		glm::vec4 pos;
		glm::vec4 col;
		glm::vec4 vel;
	};

}