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
#include <algorithm>
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
#include "R7UserInterface.h"

// Forward declare R7Editor functions used in Core::Initialize to stop circular include
namespace R7::R7Editor {
    void Initialize(GLFWwindow* window, VkInstance instance, VkDevice device, VkPhysicalDevice physicalDevice, uint32_t queueFamily, VkQueue queue, VkCommandPool commandPool);
    void Shutdown();
}
namespace R7 {
    namespace Core
    {
        namespace Detail {
            inline std::vector<R7PanelBase*> registeredPanels;
        }
        inline void RegisterPanel(R7PanelBase* panel) {
            if (panel) Detail::registeredPanels.push_back(panel);
        }
        inline void UnregisterPanel(R7PanelBase* panel) {
            auto& panels = Detail::registeredPanels;
            panels.erase(std::remove(panels.begin(), panels.end(), panel), panels.end());
        }
            for (R7PanelBase* panel : Detail::registeredPanels) {
                if (panel) {
                    panel->Render(ui);
                }
            }
        }
        inline void Initialize(GLFWwindow* window,
            VkInstance instance,
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            uint32_t queueFamily,
            VkQueue queue,
            VkCommandPool commandPool) {
            if (Font::GetCachedFont().isLoaded) {
                std::cout << "[Core] Default font ready: " << Font::s_defaultFontPath << "\n";
            }
            else {
                std::cerr << "[Core Warning] Failed to load default font from: " << Font::s_defaultFontPath << "\n";
            }

            R7::R7Backend::Init();

#ifdef R7_EDITOR
            R7::R7Editor::Initialize(window, instance, device, physicalDevice,
                queueFamily, queue, commandPool);
#endif
        }
        inline void Update() {
            R7::R7Backend::Update();
        }
        inline void DestroyBackend() {
            R7::R7Backend::Shutdown();
        }
        inline void DestroyEditor() {
#ifdef R7_EDITOR
            R7::R7Editor::Shutdown();
#endif
        }
        inline void Cleanup() {
            DestroyEditor();
            DestroyBackend();
        }
    }
}