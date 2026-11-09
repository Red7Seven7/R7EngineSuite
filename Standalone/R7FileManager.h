#pragma once
// ==========================================================
// R7FileManager - Modern C++23 Thread-Safe File Utility Library
// & Embedded File Manager UI Panel
// Copyright (c) 2026 RedSeven (Licensed under the MIT License)
// ==========================================================

#if defined(_MSVC_LANG)
#  if _MSVC_LANG < 202004L   
#    error "R7FileManager requires C++23 or a newer standard."
#  endif
#else
#  if __cplusplus < 202302L
#    error "R7FileManager requires C++23 or a newer standard."
#  endif
#endif

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <expected>
#include <mutex>
#include <queue>
#include <thread>
#include <future>
#include <atomic>
#include <iostream>
#include <glm/glm.hpp>
#include <volk.h>
#include <GLFW/glfw3.h>
#include "R7UserInterface.h"

namespace R7 {

    enum class FileError {
        Success = 0,
        NotFound,
        PermissionDenied,
        OpenFailed,
        ReadFailed,
        WriteFailed,
        InvalidParameter,
        MagicMismatch
    };

    class AssetWriter {
    public:
        AssetWriter() = default;
        ~AssetWriter() { Close(); }

        AssetWriter(const AssetWriter&) = delete;
        AssetWriter& operator=(const AssetWriter&) = delete;
        AssetWriter(AssetWriter&&) noexcept = default;
        AssetWriter& operator=(AssetWriter&&) noexcept = default;

        std::expected<void, FileError> OpenWithHeader(const std::filesystem::path& filepath, uint32_t magicNumber, uint32_t version = 1) {
            auto openRes = Open(filepath);
            if (!openRes) return openRes;
            auto writeMagic = Write(magicNumber);
            if (!writeMagic) return std::unexpected(FileError::WriteFailed);
            auto writeVersion = Write(version);
            if (!writeVersion) return std::unexpected(FileError::WriteFailed);
            return {};
        }

        std::expected<void, FileError> Open(const std::filesystem::path& filepath) {
            std::lock_guard<std::mutex> lock(m_mutex);
            std::error_code ec;
            if (filepath.has_parent_path() && !std::filesystem::exists(filepath.parent_path(), ec)) {
                std::filesystem::create_directories(filepath.parent_path(), ec);
            }

            m_file.open(filepath, std::ios::binary | std::ios::out | std::ios::trunc);
            if (!m_file.is_open()) return std::unexpected(FileError::OpenFailed);
            return {};
        }

        template <typename T>
        std::expected<void, FileError> Write(const T& data) {
            static_assert(std::is_trivial_v<T>, "AssetWriter::Write requires trivial/POD types");
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_file.is_open()) return std::unexpected(FileError::WriteFailed);

            m_file.write(reinterpret_cast<const char*>(&data), sizeof(T));
            return m_file.good() ? std::expected<void, FileError>{} : std::unexpected(FileError::WriteFailed);
        }

        template <typename T>
        std::expected<void, FileError> WriteVector(const std::vector<T>& vec) {
            static_assert(std::is_trivial_v<T>, "AssetWriter::WriteVector requires trivial/POD types");
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_file.is_open()) return std::unexpected(FileError::WriteFailed);

            size_t size = vec.size();
            Write_Unsafe(size);
            if (size > 0) {
                m_file.write(reinterpret_cast<const char*>(vec.data()), sizeof(T) * size);
            }
            return m_file.good() ? std::expected<void, FileError>{} : std::unexpected(FileError::WriteFailed);
        }

        std::expected<void, FileError> WriteString(const std::string& str) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_file.is_open()) return std::unexpected(FileError::WriteFailed);

            uint32_t length = static_cast<uint32_t>(str.size());
            Write_Unsafe(length);
            m_file.write(str.data(), length);
            return m_file.good() ? std::expected<void, FileError>{} : std::unexpected(FileError::WriteFailed);
        }

        void Close() {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_file.is_open()) m_file.close();
        }

    private:
        template <typename T>
        void Write_Unsafe(const T& data) {
            m_file.write(reinterpret_cast<const char*>(&data), sizeof(T));
        }

        std::ofstream m_file;
        std::mutex m_mutex;
    };

    class R7FileManager {
    public:
        R7FileManager() = delete;

        static std::expected<void, FileError> WriteFile(const std::filesystem::path& filepath, const void* data, size_t sizeBytes) {
            std::error_code ec;
            if (filepath.has_parent_path() && !std::filesystem::exists(filepath.parent_path(), ec)) {
                std::filesystem::create_directories(filepath.parent_path(), ec);
            }

            std::ofstream file(filepath, std::ios::binary | std::ios::trunc);
            if (!file.is_open()) return std::unexpected(FileError::OpenFailed);

            file.write(reinterpret_cast<const char*>(data), sizeBytes);
            return file.good() ? std::expected<void, FileError>{} : std::unexpected(FileError::WriteFailed);
        }

        static std::expected<void, FileError> WriteTextFile(const std::filesystem::path& filepath, const std::string& text) {
            std::error_code ec;
            if (filepath.has_parent_path() && !std::filesystem::exists(filepath.parent_path(), ec)) {
                std::filesystem::create_directories(filepath.parent_path(), ec);
            }

            std::ofstream file(filepath, std::ios::trunc);
            if (!file.is_open()) return std::unexpected(FileError::OpenFailed);

            file << text;
            return file.good() ? std::expected<void, FileError>{} : std::unexpected(FileError::WriteFailed);
        }

        static std::expected<std::vector<char>, FileError> ReadFile(const std::filesystem::path& filepath) {
            std::error_code ec;
            if (!std::filesystem::exists(filepath, ec)) return std::unexpected(FileError::NotFound);

            std::ifstream file(filepath, std::ios::binary | std::ios::ate);
            if (!file.is_open()) return std::unexpected(FileError::OpenFailed);

            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<char> outBuffer(size);
            if (!file.read(outBuffer.data(), size)) return std::unexpected(FileError::ReadFailed);
            return outBuffer;
        }

        static std::expected<uint32_t, FileError> ValidateHeader(const std::filesystem::path& filepath, uint32_t expectedMagic) {
            auto fileData = ReadFile(filepath);
            if (!fileData) return std::unexpected(fileData.error());

            if (fileData->size() < sizeof(uint32_t) * 2) return std::unexpected(FileError::ReadFailed);

            uint32_t fileMagic = *reinterpret_cast<const uint32_t*>(fileData->data());
            uint32_t fileVersion = *reinterpret_cast<const uint32_t*>(fileData->data() + sizeof(uint32_t));

            if (fileMagic != expectedMagic) return std::unexpected(FileError::MagicMismatch);
            return fileVersion;
        }

        static bool Remove(const std::filesystem::path& filepath) {
            std::error_code ec;
            return std::filesystem::remove(filepath, ec);
        }

        static bool Exists(const std::filesystem::path& filepath) {
            return std::filesystem::exists(filepath);
        }

        static std::vector<std::filesystem::path> GetFilesWithExtension(const std::filesystem::path& directory, const std::string& extension) {
            std::vector<std::filesystem::path> results;
            std::error_code ec;
            if (!std::filesystem::exists(directory, ec)) return results;

            for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
                if (entry.is_regular_file() && entry.path().extension() == extension) {
                    results.push_back(entry.path());
                }
            }
            return results;
        }
    };

    // ==========================================
    // Embedded FileManagerPanel (Moved directly into R7FileManager.h)
    // ==========================================
    namespace UI_Panel_Atlas {
        static constexpr glm::vec2 ArrowUpMin = { 0.1875f, 0.125f };
        static constexpr glm::vec2 ArrowUpMax = { 0.2500f, 0.1875f };
        static constexpr glm::vec2 FileIconMin = { 0.0000f, 0.0625f };
        static constexpr glm::vec2 FileIconMax = { 0.0625f, 0.1250f };
        static constexpr glm::vec2 PanelBgMin = { 0.7500f, 0.0000f };
        static constexpr glm::vec2 PanelBgMax = { 0.8125f, 0.0625f };
    }

    struct FileEntry {
        std::filesystem::path path;
        std::string name;
        bool isDirectory;
        uintmax_t fileSize;
    };

    template<float SizeX = 700.0f, float SizeY = 500.0f>
    class FileManagerPanel : public R7SizedPanel<SizeX, SizeY> {
    public:
        FileManagerPanel(VkDescriptorSet atlasDescriptor = VK_NULL_HANDLE, const std::filesystem::path& initialPath = std::filesystem::current_path())
            : R7SizedPanel<SizeX, SizeY>(atlasDescriptor), m_currentPath(initialPath) {
            RefreshDirectory();
        }

        void SetCurrentPath(const std::filesystem::path& path) {
            if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
                m_currentPath = path;
                m_selectedIndex = -1;
                m_scrollOffset = 0;
                RefreshDirectory();
            }
        }

        const std::filesystem::path& GetSelectedFile() const { return m_selectedPath; }

        virtual void Render(R7UserInterface& ui) override {
            if (this->m_texture != VK_NULL_HANDLE) {
                ui.DrawImage(this->m_posX, this->m_posY, SizeX, SizeY, this->m_texture,
                    { 0.85f, 0.88f, 0.92f, 0.98f }, UI_Panel_Atlas::PanelBgMin, UI_Panel_Atlas::PanelBgMax);
            }
            else {
                ui.DrawRect(this->m_posX, this->m_posY, SizeX, SizeY, { 0.12f, 0.14f, 0.18f, 0.95f });
            }

            ui.DrawRect(this->m_posX, this->m_posY, SizeX, 32.0f, { 0.08f, 0.10f, 0.12f, 1.0f });
            ui.DrawString(this->m_posX + 12.0f, this->m_posY + 8.0f, 1.0f, "R7 FILE MANAGER", VK_NULL_HANDLE, { 0.9f, 0.9f, 0.95f, 1.0f });

            float toolbarY = this->m_posY + 36.0f;
            ui.DrawRect(this->m_posX + 8.0f, toolbarY, SizeX - 16.0f, 32.0f, { 0.18f, 0.20f, 0.24f, 1.0f });

            if (ui.DrawButton(this->m_posX + 12.0f, toolbarY + 4.0f, 24.0f, 24.0f, "", { 0.25f, 0.28f, 0.35f, 1.0f })) {
                if (m_currentPath.has_parent_path()) {
                    SetCurrentPath(m_currentPath.parent_path());
                }
            }
            ui.DrawImage(this->m_posX + 16.0f, toolbarY + 8.0f, 16.0f, 16.0f, this->m_texture,
                { 1.0f, 1.0f, 1.0f, 1.0f }, UI_Panel_Atlas::ArrowUpMin, UI_Panel_Atlas::ArrowUpMax);

            std::string pathStr = m_currentPath.string();
            if (pathStr.length() > 55) {
                pathStr = "..." + pathStr.substr(pathStr.length() - 52);
            }
            ui.DrawRect(this->m_posX + 44.0f, toolbarY + 4.0f, SizeX - 60.0f, 24.0f, { 0.10f, 0.11f, 0.14f, 1.0f });
            ui.DrawString(this->m_posX + 50.0f, toolbarY + 8.0f, 0.85f, pathStr, VK_NULL_HANDLE, { 0.7f, 0.85f, 1.0f, 1.0f });

            float listX = this->m_posX + 12.0f;
            float listY = toolbarY + 38.0f;
            float listW = SizeX - 24.0f;
            float listH = SizeY - 120.0f;

            ui.DrawRect(listX, listY, listW, listH, { 0.08f, 0.09f, 0.11f, 1.0f });

            const float rowHeight = 26.0f;
            int maxVisibleRows = static_cast<int>(listH / rowHeight);

            for (int i = 0; i < maxVisibleRows; ++i) {
                int entryIndex = m_scrollOffset + i;
                if (entryIndex >= static_cast<int>(m_entries.size())) break;

                const auto& entry = m_entries[entryIndex];
                float rowY = listY + (i * rowHeight);
                bool isSelected = (entryIndex == m_selectedIndex);

                glm::vec4 rowBg = isSelected ? glm::vec4(0.2f, 0.35f, 0.55f, 1.0f)
                    : ((i % 2 == 0) ? glm::vec4(0.12f, 0.13f, 0.16f, 1.0f) : glm::vec4(0.10f, 0.11f, 0.14f, 1.0f));

                if (ui.DrawButton(listX + 2.0f, rowY + 2.0f, listW - 4.0f, rowHeight - 2.0f, "", rowBg)) {
                    m_selectedIndex = entryIndex;
                    m_selectedPath = entry.path;
                    if (entry.isDirectory) {
                        SetCurrentPath(entry.path);
                        return;
                    }
                }

                glm::vec4 iconTint = entry.isDirectory ? glm::vec4(1.0f, 0.8f, 0.3f, 1.0f) : glm::vec4(0.6f, 0.75f, 0.9f, 1.0f);
                ui.DrawImage(listX + 8.0f, rowY + 5.0f, 16.0f, 16.0f, this->m_texture, iconTint, UI_Panel_Atlas::FileIconMin, UI_Panel_Atlas::FileIconMax);

                std::string label = entry.name + (entry.isDirectory ? "/" : "");
                if (label.length() > 40) label = label.substr(0, 37) + "...";
                ui.DrawString(listX + 32.0f, rowY + 6.0f, 0.9f, label, VK_NULL_HANDLE, { 0.9f, 0.9f, 0.9f, 1.0f });

                if (!entry.isDirectory) {
                    ui.DrawString(listX + listW - 100.0f, rowY + 6.0f, 0.8f, FormatSize(entry.fileSize), VK_NULL_HANDLE, { 0.5f, 0.6f, 0.7f, 1.0f });
                }
            }

            float footerY = this->m_posY + SizeY - 36.0f;
            ui.DrawRect(this->m_posX + 12.0f, footerY, SizeX - 24.0f, 28.0f, { 0.15f, 0.17f, 0.20f, 1.0f });

            std::string statusText = (m_selectedIndex >= 0 && m_selectedIndex < static_cast<int>(m_entries.size()))
                ? "Selected: " + m_entries[m_selectedIndex].name
                : std::to_string(m_entries.size()) + " items";

            ui.DrawString(this->m_posX + 20.0f, footerY + 7.0f, 0.85f, statusText, VK_NULL_HANDLE, { 0.7f, 0.7f, 0.75f, 1.0f });

            if (ui.DrawButton(this->m_posX + SizeX - 116.0f, footerY + 3.0f, 100.0f, 22.0f, "", { 0.2f, 0.35f, 0.3f, 1.0f })) {
                InspectSelectedAsset();
            }
            ui.DrawString(this->m_posX + SizeX - 102.0f, footerY + 7.0f, 0.80f, "Load Asset", VK_NULL_HANDLE, { 0.9f, 0.95f, 0.9f, 1.0f });
        }

    private:
        void RefreshDirectory() {
            m_entries.clear();
            std::error_code ec;
            if (!std::filesystem::exists(m_currentPath, ec)) return;

            for (const auto& entry : std::filesystem::directory_iterator(m_currentPath, ec)) {
                FileEntry e;
                e.path = entry.path();
                e.name = entry.path().filename().string();
                e.isDirectory = entry.is_directory(ec);
                e.fileSize = e.isDirectory ? 0 : entry.file_size(ec);

                if (e.isDirectory) m_entries.insert(m_entries.begin(), e);
                else m_entries.push_back(e);
            }
        }

        void InspectSelectedAsset() {
            if (m_selectedIndex < 0 || m_selectedIndex >= static_cast<int>(m_entries.size())) return;
            const auto& entry = m_entries[m_selectedIndex];
            if (entry.isDirectory) return;

            auto versionRes = R7::R7FileManager::ValidateHeader(entry.path, 0x52374153);
            if (versionRes) {
                std::cout << "[FileManagerPanel] Valid R7 Asset header found. Version: " << versionRes.value() << "\n";
            }
            else {
                std::cout << "[FileManagerPanel] Standard file selected or magic mismatch.\n";
            }
        }

        std::string FormatSize(uintmax_t bytes) const {
            if (bytes < 1024) return std::to_string(bytes) + " B";
            if (bytes < 1024 * 1024) return std::to_string(bytes / 1024) + " KB";
            return std::to_string(bytes / (1024 * 1024)) + " MB";
        }

        std::filesystem::path m_currentPath;
        std::filesystem::path m_selectedPath;
        std::vector<FileEntry> m_entries;
        int m_selectedIndex = -1;
        int m_scrollOffset = 0;
    };

} // namespace R7