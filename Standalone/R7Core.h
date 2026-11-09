#pragma once
#include <vector>
#include <string>
#include <fstream> 
#include <iostream>
#include <array>
#include <atomic>
#include <type_traits>
#include <expected>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/component_wise.hpp>
#include <R7DataStream.h>
#include "MeshCore.h"
#include "VulkanContext.h"
#include <R7FileManager.h> 
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include "R7WorldFramework.h"
#include "R7Backend.h"
#include "R7Font.h"

// Must come after Core::Font: the editor uses Font::GetCachedFont().
#ifdef R7_EDITOR
#include "R7Editor.h"
#endif

namespace R7 {
    namespace Core
    {
        inline void Initialize(GLFWwindow* window,
            VkInstance instance,
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            uint32_t queueFamily,
            VkQueue queue,
            VkCommandPool commandPool) {
            // Prime the font cache (also avoids reading the file twice).
            if (Font::GetCachedFont().isLoaded) {
                std::cout << "[Core] Default font ready: " << Font::s_defaultFontPath << "\n";
            }
            else {
                std::cerr << "[Core Warning] Failed to load default font from: " << Font::s_defaultFontPath << "\n";
            }

            R7::R7Backend::Init();

#ifdef R7_EDITOR
            // Without this call the editor's initialized flag stays false and
            // BeginFrame / RenderUI / EndFrame all return early.
            R7::R7Editor::Initialize(window, instance, device, physicalDevice,
                queueFamily, queue, commandPool);
#endif
        }

        inline void Update() {
            R7::R7Backend::Update(); // update backend chain
        }

        inline void DestroyBackend() {
            R7::R7Backend::Shutdown();
        }

        inline void DestroyEditor() {
#ifdef R7_EDITOR
            R7::R7Editor::Shutdown(); // no-op if it was never initialized
#endif
        }

        // Editor first: it holds raw pointers into the backend's worlds.
        inline void Cleanup() {
            DestroyEditor();
            DestroyBackend();
        }
    }
}