#pragma once
// ==========================================================
// R7FileManager - Modern C++23 Thread-Safe File Utility Library
// Copyright (c) 2026 RedSeven (Licensed under the MIT License)
// ==========================================================

// C++23 enforced for <future> and <expected>
#if defined(_MSVC_LANG)
#  if _MSVC_LANG < 202004L   // /std:c++latest on MSVC
#    error "R7FileManager requires C++23 or a newer standard. Use /std:c++latest (or /std:c++23preview)."
#  endif
#else
#  if __cplusplus < 202302L
#    error "R7FileManager requires C++23 or a newer standard. Please use -std=c++23 (or higher)."
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

namespace R7 {

    enum class FileError {
        Success = 0,
        NotFound,
        PermissionDenied,
        OpenFailed,
        ReadFailed,
        WriteFailed,
        InvalidParameter,
        MagicMismatch // Added for magic signature validation failures
    };

    /*
    AssetWriter
    thread-safe stream writer with Magic Header support
    */
    class AssetWriter {
    public:
        AssetWriter() = default;
        ~AssetWriter() { Close(); }

        // Prevent stream corruption via copying
        AssetWriter(const AssetWriter&) = delete;
        AssetWriter& operator=(const AssetWriter&) = delete;

        // Allow moving
        AssetWriter(AssetWriter&&) noexcept = default;
        AssetWriter& operator=(AssetWriter&&) noexcept = default;

        // Open and automatically write a 4-byte Magic Signature and Format Version header
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
            if (!m_file.is_open()) {
                return std::unexpected(FileError::OpenFailed);
            }
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
            if (m_file.is_open()) {
                m_file.close();
            }
        }

    private:
        template <typename T>
        void Write_Unsafe(const T& data) {
            m_file.write(reinterpret_cast<const char*>(&data), sizeof(T));
        }

        std::ofstream m_file;
        std::mutex m_mutex;
    };

    /* R7FileManager
    * Async capable file utility library with header validation support.
    */
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
            if (!std::filesystem::exists(filepath, ec)) {
                return std::unexpected(FileError::NotFound);
            }

            std::ifstream file(filepath, std::ios::binary | std::ios::ate);
            if (!file.is_open()) return std::unexpected(FileError::OpenFailed);

            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<char> outBuffer(size);
            if (!file.read(outBuffer.data(), size)) {
                return std::unexpected(FileError::ReadFailed);
            }
            return outBuffer;
        }

        // Validate 4-byte magic signature and extract file format version
        static std::expected<uint32_t, FileError> ValidateHeader(const std::filesystem::path& filepath, uint32_t expectedMagic) {
            auto fileData = ReadFile(filepath);
            if (!fileData) return std::unexpected(fileData.error());

            if (fileData->size() < sizeof(uint32_t) * 2) {
                return std::unexpected(FileError::ReadFailed);
            }

            uint32_t fileMagic = *reinterpret_cast<const uint32_t*>(fileData->data());
            uint32_t fileVersion = *reinterpret_cast<const uint32_t*>(fileData->data() + sizeof(uint32_t));

            if (fileMagic != expectedMagic) {
                return std::unexpected(FileError::MagicMismatch);
            }

            return fileVersion; // Returns the version number if valid
        }

        static std::expected<void, FileError> AppendToFile(const std::filesystem::path& filepath, const void* data, size_t sizeBytes) {
            std::ofstream file(filepath, std::ios::binary | std::ios::app);
            if (!file.is_open()) return std::unexpected(FileError::OpenFailed);

            file.write(reinterpret_cast<const char*>(data), sizeBytes);
            return file.good() ? std::expected<void, FileError>{} : std::unexpected(FileError::WriteFailed);
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

        struct ReadResult {
            std::filesystem::path filepath;
            std::expected<std::vector<char>, FileError> result;
        };

        static std::vector<std::future<ReadResult>> ReadFilesParallel(
            const std::vector<std::filesystem::path>& filepaths)
        {
            std::vector<std::future<ReadResult>> futures;
            futures.reserve(filepaths.size());

            for (const auto& path : filepaths) {
                futures.push_back(std::async(std::launch::async, [path]() {
                    ReadResult res;
                    res.filepath = path;
                    res.result = ReadFile(path);
                    return res;
                }));
            }

            return futures;
        }

        class AsyncWriterQueue {
        public:
            AsyncWriterQueue() : m_running(true) {
                m_worker = std::thread(&AsyncWriterQueue::ProcessQueue, this);
            }
            ~AsyncWriterQueue() {
                Stop();
            }
            void EnqueueWrite(std::filesystem::path filepath, std::vector<char> data) {
                {
                    std::lock_guard<std::mutex> lock(m_queueMutex);
                    m_tasks.push({ std::move(filepath), std::move(data) });
                }
                m_cv.notify_one();
            }
            void Stop() {
                {
                    std::lock_guard<std::mutex> lock(m_queueMutex);
                    m_running = false;
                }
                m_cv.notify_all();
                if (m_worker.joinable()) {
                    m_worker.join();
                }
            }
        private:
            struct WriteTask {
                std::filesystem::path filepath;
                std::vector<char> data;
            };
            void ProcessQueue() {
                while (true) {
                    WriteTask task;
                    {
                        std::unique_lock<std::mutex> lock(m_queueMutex);
                        m_cv.wait(lock, [this] { return !m_tasks.empty() || !m_running; });

                        if (!m_running && m_tasks.empty()) break;

                        task = std::move(m_tasks.front());
                        m_tasks.pop();
                    }

                    R7FileManager::WriteFile(task.filepath, task.data.data(), task.data.size());
                }
            }
            std::queue<WriteTask> m_tasks;
            std::mutex m_queueMutex;
            std::condition_variable m_cv;
            std::atomic<bool> m_running;
            std::thread m_worker;
        };

        static AsyncWriterQueue& GetAsyncWriter() {
            static AsyncWriterQueue instance;
            return instance;
        }
    };

} // namespace R7