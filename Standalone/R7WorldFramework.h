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
#include <GLFW/glfw3.h>
#include "Meshcore.h"


namespace R7 {
    namespace WorldFramework {
        template <std::size_t MaxDynamic = 16, std::size_t MaxStatic = 32>
        class World;
        class Entity;
        class StaticEntity;
        class DynamicEntity;

        struct Transform
        {
            glm::vec3 Loc;
            glm::quat Rot;
            glm::vec3 Scale;

            inline glm::mat4 GetMatrix() const
            {
                glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), Loc);
                glm::mat4 rotationMatrix = glm::mat4_cast(Rot);
                glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), Scale);

                return translationMatrix * rotationMatrix * scaleMatrix;
            }
        };

        class Player {};
        class PlayerPackage {};

        class Entity
        {
            template <std::size_t MaxDynamic, std::size_t MaxStatic>
            friend class World;

        public:
            Entity(World<>* ownerWorld, const Transform& spawnTransform)
                : OwnerWorld(ownerWorld), transform(spawnTransform)
            {
            }

            virtual ~Entity() = default;

            World<>* GetWorld() const { return OwnerWorld; }

            const Transform& GetTransform() const { return transform; }
            Transform& GetTransform() { return transform; }

            virtual void UpdateTransform() {}
            virtual void UpdateRender() {}

        protected:
            Transform transform;
            World<>* OwnerWorld = nullptr;
            std::size_t worldIndex = static_cast<std::size_t>(-1);
        };

        class StaticEntity : public Entity {
        public:
            StaticEntity(World<>* ownerWorld, const Transform& spawnTransform, StaticMesh* mesh = nullptr)
                : Entity(ownerWorld, spawnTransform), Mesh(mesh)
            {
                CachedMatrix = spawnTransform.GetMatrix();
            }

            StaticMesh* Mesh = nullptr;
            glm::mat4 CachedMatrix = glm::mat4(1.0f);

        private:
            void UpdateTransform() override {}
            void UpdateRender() override {}
        };

        class DynamicEntity : public Entity {
        public:
            DynamicEntity(World<>* ownerWorld, const Transform& spawnTransform, SkinnedMesh* mesh = nullptr)
                : Entity(ownerWorld, spawnTransform), Mesh(mesh)
            {
            }

            SkinnedMesh* Mesh = nullptr;

        private:
            void UpdateTransform() override {}
            void UpdateRender() override {}
        };

        template <std::size_t MaxDynamic, std::size_t MaxStatic>
        class World
        {
        public:
            struct Slot {
                std::atomic<Entity*> entity{ nullptr };
                std::atomic<bool> isOccupied{ false };
            };

            World()
                : maxDynamicCapacity(MaxDynamic), maxStaticCapacity(MaxStatic)
            {
                slots = new Slot[maxDynamicCapacity]();
                staticEntities.reserve(maxStaticCapacity);
            }

            virtual ~World()
            {
                for (std::size_t i = 0; i < maxDynamicCapacity; ++i) {
                    if (slots[i].isOccupied.load(std::memory_order_acquire)) {
                        Entity* ent = slots[i].entity.load(std::memory_order_relaxed);
                        delete ent;
                    }
                }
                delete[] slots;

                for (StaticEntity* ent : staticEntities) {
                    delete ent;
                }
            }

            template <typename T, typename... Args>
            T* CreateEntity(const Transform& spawnTransform, Args&&... args)
            {
                static_assert(std::is_base_of_v<DynamicEntity, T>, "T must derive from DynamicEntity");

                for (std::size_t i = 0; i < maxDynamicCapacity; ++i) {
                    bool expected = false;
                    if (slots[i].isOccupied.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                        T* newEntity = new T(this, spawnTransform, std::forward<Args>(args)...);
                        newEntity->worldIndex = i;
                        slots[i].entity.store(newEntity, std::memory_order_release);
                        return newEntity;
                    }
                }
                return nullptr;
            }

            template <typename T, typename... Args>
            T* CreateStaticEntity(const Transform& spawnTransform, Args&&... args)
            {
                static_assert(std::is_base_of_v<StaticEntity, T>, "T must derive from StaticEntity");
                if (staticEntities.size() >= maxStaticCapacity) return nullptr;

                T* newEntity = new T(this, spawnTransform, std::forward<Args>(args)...);
                staticEntities.push_back(newEntity);
                return newEntity;
            }

            void DestroyEntity(Entity* entity)
            {
                if (!entity) return;

                std::size_t idx = entity->worldIndex;
                if (idx >= maxDynamicCapacity) return;

                Entity* expectedEntity = entity;
                if (slots[idx].entity.compare_exchange_strong(expectedEntity, nullptr, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    slots[idx].isOccupied.store(false, std::memory_order_release);
                    delete entity;
                }
            }

            void UpdateWorld()
            {
                if (IsDormant) return;

                for (std::size_t i = 0; i < maxDynamicCapacity; ++i) {
                    if (slots[i].isOccupied.load(std::memory_order_acquire)) {
                        Entity* ent = slots[i].entity.load(std::memory_order_relaxed);
                        if (ent) {
                            ent->UpdateTransform();
                            ent->UpdateRender();
                        }
                    }
                }
            }

            Slot* slots = nullptr;
            std::vector<StaticEntity*> staticEntities;
            std::size_t maxDynamicCapacity = 0;
            std::size_t maxStaticCapacity = 0;
            std::uint8_t depth = 0;
        private:
            bool IsDormant = true;
        };
        

        class TinyWorld : public World<8, 16> {};
        class SmallWorld : public World<16, 32> {};
        class RegualrWorld : public World<16, 32> {};
        class MediumWorld : public World<32, 64> {};
        class LargeWorld : public World<64, 128> {};
        class ColossalWorld : public World<128, 256> {};


    };
};