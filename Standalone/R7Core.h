#pragma once

#include "VulkanContext.h"
#include <R7FileManager.h> 
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include "R7Font.h"
#include "R7UserInterface.h"
import std;
import R7Data;
import R7WorldFramework;
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
        inline void RenderPanels(R7UserInterface& ui) {
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


#ifdef R7_EDITOR
            R7::R7Editor::Initialize(window, instance, device, physicalDevice,
                queueFamily, queue, commandPool);
#endif
        }
        inline void Update() {
        }
        inline void DestroyEditor() {
#ifdef R7_EDITOR
            R7::R7Editor::Shutdown();
#endif
        }
        inline void Cleanup() {
            DestroyEditor();
        }
    }
}