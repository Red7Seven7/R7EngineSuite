#pragma once

#include <atomic>
#include <array>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <mutex>
#include <type_traits>
#include <filesystem>

// Detect x86 / x64 architectures,
// X86 should not matter.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
constexpr std::memory_order kStoreOrder = std::memory_order_relaxed;
constexpr std::memory_order kLoadOrder = std::memory_order_relaxed;
#else
    // Weakly-ordered architectures (ARM, Apple Silicon, mobile, consoles)
    // Arm or other silicon, found out it can cause some issues.
constexpr std::memory_order kStoreOrder = std::memory_order_release;
constexpr std::memory_order kLoadOrder = std::memory_order_acquire;
#endif

// Check for glm dependency at compile time
#if __has_include("glm/glm.hpp")
#include "glm/glm.hpp"


namespace R7 {
    //note: this uses data tearing intentionally.
    //it may eat half-baked frames or stale frames, this is by design.
    //only use it for graphics or anything that gracefully accepts
    //reading half-updated values for determinism.


    struct Matrix4x4 {
        float m[4][4];
    };
    struct RenderPacket {
        Matrix4x4 transform;
        unsigned int objectId;
    };
    class RipStream {
    //Data tearing optimised stream with architecture-specific memory ordering
    
    public:
        RipStream() {
            m_writeIndex.store(0, std::memory_order_relaxed);

            for (int i = 0; i < 3; ++i) {
                m_slots[i].objectId = 0;
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        m_slots[i].transform.m[row][col] = (row == col) ? 1.0f : 0.0f;
                    }
                }
            }
        }

        inline void WriteSimulationData(const Matrix4x4& newMatrix, unsigned int id) {
            size_t currentWrite = m_writeIndex.load(std::memory_order_relaxed);
            size_t nextWrite = (currentWrite + 1) % 3;

            // Direct write allowing intentional tearing
            m_slots[currentWrite].transform = newMatrix;
            m_slots[currentWrite].objectId = id;

            m_writeIndex.store(nextWrite, kStoreOrder);
        }

        inline RenderPacket ReadDataForGPU() const {
            size_t latest = m_writeIndex.load(kLoadOrder);
            size_t readIdx = (latest + 3 - 1) % 3;
            return m_slots[readIdx];
        }

    private:
        //alignas 64 prevents false sharing across different cores.
        alignas(64) std::atomic<size_t> m_writeIndex{ 0 };
        alignas(64) RenderPacket m_slots[3];
    };
    template <typename T, size_t Capacity = 4>
    class TearingRingBuffer {
     //Data tearing optimised ring buffer with architecture-specific memory ordering.
    public:
        TearingRingBuffer() {
            m_writeIndex.store(0, std::memory_order_relaxed);
        }
        void Write(const T& packet) {
            size_t currentWrite = m_writeIndex.load(std::memory_order_relaxed);
            size_t nextWrite = (currentWrite + 1) % Capacity;

            m_slots[currentWrite] = packet;
            m_writeIndex.store(nextWrite, kStoreOrder);
        }
        T ReadLatest() const {
            size_t latest = m_writeIndex.load(kLoadOrder);
            size_t readIdx = (latest + Capacity - 1) % Capacity;
            return m_slots[readIdx];
        }
    private:
        alignas(64) std::atomic<size_t> m_writeIndex;
        alignas(64) std::array<T, Capacity> m_slots;
    };
    class AssetStream {
    //Thread-safe sequential binary stream for loading worlds, meshes, and serialized assets.
    //Uses internal mutex locking to allow safe concurrent reads from multiple worker threads.
    public:
        bool Open(const std::string& filepath) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_file.open(filepath, std::ios::binary | std::ios::ate);
            if (!m_file.is_open()) return false;

            m_fileSize = static_cast<size_t>(m_file.tellg());
            m_file.seekg(0, std::ios::beg);
            m_buffer.resize(m_fileSize);

            if (!m_file.read(m_buffer.data(), m_fileSize)) {
                return false;
            }
            m_file.close();
            m_offset = 0;
            return true;
        }
        template <typename T>
        bool Read(T& outData) {
            static_assert(std::is_trivial_v<T>, "AssetStream::Read requires trivial/POD types");
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_offset + sizeof(T) > m_fileSize) return false;

            std::memcpy(&outData, m_buffer.data() + m_offset, sizeof(T));
            m_offset += sizeof(T);
            return true;
        }
        template <typename T>
        bool ReadVector(std::vector<T>& outVector, size_t count) {
            static_assert(std::is_trivial_v<T>, "AssetStream::ReadVector requires trivial/POD types");
            std::lock_guard<std::mutex> lock(m_mutex);
            size_t bytesNeeded = sizeof(T) * count;
            if (m_offset + bytesNeeded > m_fileSize) return false;

            outVector.resize(count);
            std::memcpy(outVector.data(), m_buffer.data() + m_offset, bytesNeeded);
            m_offset += bytesNeeded;
            return true;
        }
        bool HasBytesRemaining() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_offset < m_fileSize;
        }
        size_t GetPosition() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_offset;
        }
        size_t GetSize() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_fileSize;
        }

    private:
        mutable std::mutex m_mutex;
        std::ifstream m_file;
        std::vector<char> m_buffer;
        size_t m_fileSize = 0;
        size_t m_offset = 0;
    };
};

#else
#error "R7AssetManager requires glm to be included and present in your header search paths."
#endif
