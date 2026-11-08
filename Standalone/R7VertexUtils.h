#pragma once
// ==========================================================
// R7VertexUtils - Modern C++23 vertex utils using glm
// Copyright (c) 2026 RedSeven (Licensed under the MIT License)
// ==========================================================

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/component_wise.hpp>

// Check for glm dependency at compile time
#if __has_include("glm/glm.hpp")
#include <glm/glm.hpp>
#else
#error "R7VertexUtils requires glm to be included and present in your header search paths."
#error "please add https://github.com/g-truc/glm to your search paths"
#endif

namespace R7
{
    struct BaseVertex {
        glm::vec3 position;
        glm::vec2 texCoord;
    };
    struct Vertex : public BaseVertex {
        glm::vec3 normal;
        glm::vec4 tangent;
    };
    struct SkinnedVertex : public Vertex {
        glm::uvec4 boneIDs;
        glm::vec4 weights;
    };
    struct UIVertex : public BaseVertex {
        glm::vec4 color;
        //helper constructor
        UIVertex() = default;//default constructor
        UIVertex(const glm::vec3& pos, const glm::vec2& uv, const glm::vec4& col) {
            position = pos;
            texCoord = uv;
            color = col;
        }
    };
    struct BasePackedVertex
    {
        std::vector<unsigned int > indices = {};
        //gpu handles
        unsigned int VAO = 0, VBO = 0, EBO = 0;
    };
    struct PackedVertex : public BasePackedVertex
    {
        std::vector<Vertex>Vertices = {};
    };
    struct PackedSKinnedVertex :public PackedVertex
    {
        std::vector<SkinnedVertex>Vertices = {};
        glm::uvec4 boneIndices = glm::uvec4(0);//standard max 4  bone influenece per vertex
        glm::vec4 boneWeights = glm::vec4(0.0f);//weight must sum to 1.0f

    };
};
