#pragma once
// ==========================================================
// R7AssetManager - Modern C++23 Thread-Safe Standalone Asset Manager
// Copyright (c) 2026 RedSeven (Licensed under the MIT License)
// ==========================================================

// Enforce C++23 Standard Requirement
#if defined(_MSVC_LANG)
#if _MSVC_LANG < 202302L
#error "R7AssetManager requires C++23 or a newer standard. Please enable C++23 in your compiler settings."
#endif
#else
#if __cplusplus < 202302L
#error "R7AssetManager requires C++23 or a newer standard. Please use -std=c++23 (or higher)."
#endif
#endif

// Check for R7FileManager dependency at compile time
#if __has_include("R7FileManager.h")
#include "R7FileManager.h"
#else
#error "R7AssetManager requires R7FileManager.h to be included and present in your header search paths."
#endif

#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <expected>
#include <filesystem>
#include <vector>

// Automatically detect Vulkan SDK availability via __has_include
#if __has_include(<vulkan/vulkan.h>)
#include <vulkan/vulkan.h>
#define R7_HAS_VULKAN 1
#else
#define R7_HAS_VULKAN 0
#endif

namespace R7 {

    enum class AssetError {
        Success = 0,
        NotFound,
        LoadFailed,
        InvalidType,
        FileReadFailed,
        InvalidMagicMismatch,
        VersionMismatch
    };

#if R7_HAS_VULKAN
    struct R7TextureAsset {
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;

        R7TextureAsset() = default;
        ~R7TextureAsset() = default;

        // Prevent copying to ensure safe Vulkan resource ownership
        R7TextureAsset(const R7TextureAsset&) = delete;
        R7TextureAsset& operator=(const R7TextureAsset&) = delete;
        R7TextureAsset(R7TextureAsset&&) noexcept = default;
        R7TextureAsset& operator=(R7TextureAsset&&) noexcept = default;
    };
#endif

    class R7AssetManager {
    public:
        R7AssetManager(const R7AssetManager&) = delete;
        R7AssetManager& operator=(const R7AssetManager&) = delete;

        static R7AssetManager& Get() {
            static R7AssetManager instance;
            return instance;
        }

        // Check if an asset is already cached
        bool IsLoaded(const std::string& path) const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_cache.find(path) != m_cache.end();
        }

        // Generic Load Asset Template with Magic Number & Version Validation via R7FileManager
        template<typename T, typename... Args>
        std::expected<std::shared_ptr<T>, AssetError> Load(const std::string& path, uint32_t expectedMagic = 0x52374153, Args&&... args) {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = m_cache.find(path);
                if (it != m_cache.end()) {
                    auto casted = std::dynamic_pointer_cast<T>(it->second);
                    if (!casted) {
                        return std::unexpected(AssetError::InvalidType);
                    }
                    return casted;
                }
            }

            std::filesystem::path filePath(path);
            if (!R7FileManager::Exists(filePath)) {
                return std::unexpected(AssetError::NotFound);
            }

            // Leverage R7FileManager::ValidateHeader to check magic bytes & fetch version safely
            auto versionResult = R7FileManager::ValidateHeader(filePath, expectedMagic);
            if (!versionResult) {
                switch (versionResult.error()) {
                case FileError::MagicMismatch: return std::unexpected(AssetError::InvalidMagicMismatch);
                default: return std::unexpected(AssetError::FileReadFailed);
                }
            }
            uint32_t fileVersion = versionResult.value();

            // Read the raw binary file data
            auto rawData = R7FileManager::ReadFile(filePath);
            if (!rawData) {
                return std::unexpected(AssetError::FileReadFailed);
            }

            const auto& bytes = rawData.value();
            if (bytes.size() < 8) {
                return std::unexpected(AssetError::FileReadFailed);
            }

            // Strip the 8-byte header (4-byte magic + 4-byte version) to isolate asset payload
            std::vector<char> payload(bytes.begin() + 8, bytes.end());

            // Construct and initialize asset instance
            auto asset = LoadAssetFromFile<T>(path, payload, fileVersion, std::forward<Args>(args)...);
            if (!asset) {
                return std::unexpected(AssetError::LoadFailed);
            }

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_cache[path] = asset;
            }

            return asset;
        }

        // Unload a specific asset from the cache
        template<typename T>
        void Unload(const std::string& path) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_cache.erase(path);
        }

        // Clear the entire asset cache
        void Clear() {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_cache.clear();
        }

    private:
        R7AssetManager() = default;
        ~R7AssetManager() = default;

        mutable std::mutex m_mutex;
        std::unordered_map<std::string, std::shared_ptr<void>> m_cache;

        // Internal asset factory implementation hook with stripped payload and version info
        template<typename T, typename... Args>
        std::shared_ptr<T> LoadAssetFromFile(const std::string& path, const std::vector<char>& payloadData, uint32_t fileVersion, Args&&... args) {
            return std::make_shared<T>(std::forward<Args>(args)...);
        }
    };

    // wrapped namespace functions
    template<typename T, typename... Args>
    inline std::expected<std::shared_ptr<T>, AssetError> LoadAsset(const std::string& path, uint32_t expectedMagic = 0x52374153, Args&&... args) {
        return R7AssetManager::Get().Load<T>(path, expectedMagic, std::forward<Args>(args)...);
    }

    template<typename T>
    inline bool IsAssetLoaded(const std::string& path) {
        return R7AssetManager::Get().IsLoaded(path);
    }

    template<typename T>
    inline void UnloadAsset(const std::string& path) {
        R7AssetManager::Get().Unload<T>(path);
    }

} // namespace R7