#pragma once
// ==========================================================
// R7AssetManager - Thread-Safe Standalone Asset Manager (Namespace)
// Copyright (c) 2026 RedSeven (Licensed under the MIT License)
//
// Deliberately UI-free: anything (editor, renderer, core) may include this
// without creating circular dependencies. The asset browser panel lives in
// the editor.
// ==========================================================

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
#include <typeindex>
#include <type_traits>
#include <cstdint>

#if __has_include(<vulkan/vulkan.h>)
#include <vulkan/vulkan.h>
#define R7_HAS_VULKAN 1
#else
#define R7_HAS_VULKAN 0
#endif

namespace R7::AssetManager {

    enum class AssetError {
        Success = 0,
        NotFound,
        LoadFailed,
        InvalidType,
        FileReadFailed,
        InvalidMagicMismatch,
        VersionMismatch
    };

    inline constexpr uint32_t DefaultMagic = 0x52374153;

    // Shared, immutable byte buffer (fonts, shaders, any headerless file)
    using RawBytes = std::shared_ptr<const std::vector<char>>;

#if R7_HAS_VULKAN
    struct R7TextureAsset {
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;

        R7TextureAsset() = default;
        ~R7TextureAsset() = default;

        R7TextureAsset(const R7TextureAsset&) = delete;
        R7TextureAsset& operator=(const R7TextureAsset&) = delete;
        R7TextureAsset(R7TextureAsset&&) noexcept = default;
        R7TextureAsset& operator=(R7TextureAsset&&) noexcept = default;
    };
#endif

    // ------------------------------------------------------------------
    // Internal state
    // ------------------------------------------------------------------
    namespace Detail {
        struct CacheEntry {
            std::shared_ptr<void> ptr;
            std::type_index type;
            CacheEntry(std::shared_ptr<void> p, std::type_index t)
                : ptr(std::move(p)), type(t) {
            }
        };

        inline std::mutex mutex;
        inline std::unordered_map<std::string, CacheEntry> cache;

        // Returns the cached asset, InvalidType on a type clash, or an empty
        // shared_ptr (no error) when the path isn't cached.
        template<typename T>
        std::expected<std::shared_ptr<T>, AssetError> Find(const std::string& path) {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = cache.find(path);
            if (it == cache.end()) return std::shared_ptr<T>{};
            if (it->second.type != std::type_index(typeid(T))) return std::unexpected(AssetError::InvalidType);
            return std::static_pointer_cast<T>(it->second.ptr);
        }

        // Inserts unless another thread got there first; returns whichever is cached.
        template<typename T>
        std::expected<std::shared_ptr<T>, AssetError> Insert(const std::string& path, std::shared_ptr<T> asset) {
            std::lock_guard<std::mutex> lock(mutex);
            auto [it, inserted] = cache.try_emplace(
                path,
                std::static_pointer_cast<void>(asset),
                std::type_index(typeid(T)));
            if (!inserted) {
                if (it->second.type != std::type_index(typeid(T))) return std::unexpected(AssetError::InvalidType);
                return std::static_pointer_cast<T>(it->second.ptr);
            }
            return asset;
        }

        template<typename T, typename... Args>
        std::shared_ptr<T> Construct(const std::vector<char>& payload, uint32_t fileVersion, Args&&... args) {
            // Preferred: asset can build itself from the file payload.
            if constexpr (std::is_constructible_v<T, const std::vector<char>&, uint32_t, Args...>) {
                return std::make_shared<T>(payload, fileVersion, std::forward<Args>(args)...);
            }
            // Fallback: asset is built from the caller's args only (e.g. GPU resources).
            else {
                return std::make_shared<T>(std::forward<Args>(args)...);
            }
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    inline bool IsLoaded(const std::string& path) {
        std::lock_guard<std::mutex> lock(Detail::mutex);
        return Detail::cache.find(path) != Detail::cache.end();
    }

    inline void Unload(const std::string& path) {
        std::lock_guard<std::mutex> lock(Detail::mutex);
        Detail::cache.erase(path);
    }

    // Drops the cache's references. Holders of a shared_ptr keep their asset alive.
    inline void Clear() {
        std::lock_guard<std::mutex> lock(Detail::mutex);
        Detail::cache.clear();
    }

    // Loads a headered R7 asset (magic + version, 8 bytes, stripped from the payload).
    // T is built from (payload, version, args...) if it has such a constructor,
    // otherwise from (args...).
    template<typename T, typename... Args>
    std::expected<std::shared_ptr<T>, AssetError> Load(const std::string& path, uint32_t expectedMagic = DefaultMagic, Args&&... args) {
        auto cached = Detail::Find<T>(path);
        if (!cached) return std::unexpected(cached.error());
        if (cached.value()) return cached.value();

        std::filesystem::path filePath(path);
        if (!R7::R7FileManager::Exists(filePath)) return std::unexpected(AssetError::NotFound);

        auto versionResult = R7::R7FileManager::ValidateHeader(filePath, expectedMagic);
        if (!versionResult) {
            switch (versionResult.error()) {
            case R7::FileError::MagicMismatch: return std::unexpected(AssetError::InvalidMagicMismatch);
            default: return std::unexpected(AssetError::FileReadFailed);
            }
        }
        uint32_t fileVersion = versionResult.value();

        auto rawData = R7::R7FileManager::ReadFile(filePath);
        if (!rawData) return std::unexpected(AssetError::FileReadFailed);

        const auto& bytes = rawData.value();
        if (bytes.size() < 8) return std::unexpected(AssetError::FileReadFailed);

        std::vector<char> payload(bytes.begin() + 8, bytes.end());
        auto asset = Detail::Construct<T>(payload, fileVersion, std::forward<Args>(args)...);
        if (!asset) return std::unexpected(AssetError::LoadFailed);

        return Detail::Insert<T>(path, std::move(asset));
    }

    // Loads a headerless file (.ttf, .spv, ...) as a shared, cached, immutable buffer.
    inline std::expected<RawBytes, AssetError> LoadRaw(const std::string& path) {
        using Buffer = std::vector<char>;   // no const here

        auto cached = Detail::Find<Buffer>(path);
        if (!cached) return std::unexpected(cached.error());
        if (cached.value()) return RawBytes(cached.value());

        std::filesystem::path filePath(path);
        if (!R7::R7FileManager::Exists(filePath)) return std::unexpected(AssetError::NotFound);

        auto data = R7::R7FileManager::ReadFile(filePath);
        if (!data) return std::unexpected(AssetError::FileReadFailed);

        auto buffer = std::make_shared<Buffer>(std::move(data.value()));
        auto inserted = Detail::Insert<Buffer>(path, buffer);
        if (!inserted) return std::unexpected(inserted.error());
        return RawBytes(inserted.value());   // shared_ptr<vector> -> shared_ptr<const vector> is implicit
    }

} // namespace R7::AssetManager
