#pragma once
#include <vector>
#include <string>
#include <fstream> 
#include <array>
#include <atomic>
#include <type_traits>
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

namespace R7 {
    namespace Core
    {
        namespace Font
        {
            inline const std::string s_defaultFontPath = "assets/Fonts/DefaultFont.ttf";
        }

        // Thread-safe singleton accessor for the single backend instance
        inline R7::R7Backend*& GetBackendInstance() {
            static R7::R7Backend* s_engineBackend = new R7::R7Backend();
            return s_engineBackend;
        }
    }
};

#ifdef R7_EDITOR
#include "R7Editor.h"
#endif

namespace R7 {
    namespace Core
    {
#ifdef R7_EDITOR
        // Thread-safe singleton accessor for the single editor instance, injecting R7Backend via constructor
        inline R7::R7Editor*& GetEditorInstance() {
            static R7::R7Editor* s_editor = new R7::R7Editor(GetBackendInstance());
            return s_editor;
        }

        inline R7::R7Editor* GetEditor() {
            return GetEditorInstance();
        }
#endif

        inline void Initialize(GLFWwindow* window, VkInstance instance, VkDevice device, VkPhysicalDevice physicalDevice, uint32_t queueFamily, VkQueue queue, VkCommandPool commandPool) {
            R7::R7Backend*& backend = GetBackendInstance();
            if (backend) {}

            std::vector<char> fontBuffer;
            if (R7::R7FileManager::ReadFile(Font::s_defaultFontPath)) {
                std::cout << "[Core] Successfully read default font from: " << Font::s_defaultFontPath << "\n";
            }
            else {
                std::cerr << "[Core Warning] Failed to load default font from: " << Font::s_defaultFontPath << "\n";
            }
        }

        inline void Update() {
            R7::R7Backend* backend = GetBackendInstance();
            if (backend) {
                backend->Update();
            }
        }
        inline void DestroyBackend() {
            R7::R7Backend*& backend = GetBackendInstance();
            if (!backend) return;
            delete backend;
            backend = nullptr;
        }
        inline void DestroyEditor() {
#ifdef R7_EDITOR
            R7::R7Editor*& editor = GetEditorInstance();
            if (!editor) return;
            editor->Shutdown();   // no-op if it was never initialized
            delete editor;
            editor = nullptr;
#endif
        }

        inline void Cleanup() {
            DestroyEditor();
            DestroyBackend();
        }
    }
}