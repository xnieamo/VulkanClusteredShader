#pragma once

#include <vector>

#include "SceneStructs.h"

using namespace SceneStructs;

namespace std {
	template<> struct hash < Vertex > {
		size_t operator()(Vertex const& vertex) const {
			return (
				(hash<glm::vec3>()(vertex.pos) ^
				(hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
				(hash<glm::vec2>()(vertex.texCoord) << 1 ^
				(hash<glm::vec3>()(vertex.normal) << 1));
		}
	};
}

static std::vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

void loadModel(
	std::string model_path,
	std::vector<Vertex> &vertices,
	std::vector<uint32_t> &indices) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, model_path.c_str())) {
		throw std::runtime_error(err);
	}

	std::unordered_map<Vertex, int> uniqueVertices = {};

	for (const auto& shape : shapes) {
		for (const auto& index : shape.mesh.indices) {
			Vertex vertex = {};

			vertex.pos = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]
			};

			vertex.texCoord = {
				attrib.texcoords[2 * index.texcoord_index + 0],
				1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
			};

			vertex.color = { 1.0f, 1.0f, 1.0f };

			if (!attrib.normals.empty()) {
				vertex.normal = {
					attrib.normals[3 * index.normal_index + 0],
					attrib.normals[3 * index.normal_index + 1],
					attrib.normals[3 * index.normal_index + 2]
				};
			}


			if (uniqueVertices.count(vertex) == 0) {
				uniqueVertices[vertex] = vertices.size();
				vertices.push_back(vertex);
			}

			indices.push_back(uniqueVertices[vertex]);
		}
	}

	// Compute normals if they are not in the obj.
	if (attrib.normals.empty()) {
		for (int i = 0; i < indices.size(); i += 3) {
			glm::vec3 p1 = vertices[indices[i + 0]].pos;
			glm::vec3 p2 = vertices[indices[i + 1]].pos;
			glm::vec3 p3 = vertices[indices[i + 2]].pos;

			glm::vec3 e1 = p2 - p1;
			glm::vec3 e2 = p3 - p2;

			glm::vec3 N = glm::cross(e1, e2);

			N = glm::normalize(N);
			vertices[indices[i + 0]].normal = N;
			vertices[indices[i + 1]].normal = N;
			vertices[indices[i + 2]].normal = N;
		}
	}
}


