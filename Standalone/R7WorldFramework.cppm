// R7WorldFramework.ixx : primary module interface unit
module;

// ---- Global module fragment: legacy headers only (not importable as modules) ----
#include <GLFW/glfw3.h>

export module R7WorldFramework;

import std;
import R7Data;   // must export vec3f, mat4f, transformf, toMat4, inverse, determinant, etc.

export namespace R7::WorldFramework {

    class World;
    class Entity;
    class StaticEntity;
    class DynamicEntity;

    // ------------------------------------------------------------------
    // Trace types
    // ------------------------------------------------------------------
    struct TraceParams
    {
        bool traceStatic = true;          // hit static entities
        bool traceDynamic = true;         // hit dynamic entities
        const Entity* ignore = nullptr;   // who did the trace
    };

    struct TraceHit
    {
        Entity* entity = nullptr;                 // what you hit
        float distance = 0.0f;                    // world units from start to the hit point
        vec3f position{ 0.0f, 0.0f, 0.0f };       // world-space hit point
        vec3f normal{ 0.0f, 0.0f, 0.0f };         // world-space normal of the face that was hit
    };

    // ------------------------------------------------------------------
    // World sizing (runtime values, so all worlds share one type)
    // ------------------------------------------------------------------
    enum class worldtypes : short int
    {
        TinyWorld = 0,
        SmallWorld = 1,
        RegularWorld = 2,
        MediumWorld = 3,
        LargeWorld = 4,
        ColossalWorld = 5,
        TitanicWorld = 6,
        GargantuanWorld = 7,
    };

    struct WorldConfig
    {
        std::size_t maxDynamic = 16;
        std::size_t maxStatic = 32;
    };

    constexpr WorldConfig GetWorldConfig(worldtypes type)
    {
        switch (type) {
        case worldtypes::TinyWorld:        return { 8,  16 };
        case worldtypes::SmallWorld:       return { 16, 32 };
        case worldtypes::RegularWorld:     return { 16, 32 };
        case worldtypes::MediumWorld:      return { 32, 64 };
        case worldtypes::LargeWorld:       return { 32, 128 };
        case worldtypes::ColossalWorld:    return { 32, 256 };
        case worldtypes::TitanicWorld:     return { 64, 512 };
        case worldtypes::GargantuanWorld:  return { 64, 1024 };
        }
        return {};
    }

    // ------------------------------------------------------------------
    // Player (placeholders, to be filled in)
    // ------------------------------------------------------------------
    class Player
    /* accepts player input; basically the player themselves. */
    {
    };

    class PlayerPackage
    /* defined by devs; contains everything owned by a player. */
    {
    };

    // ------------------------------------------------------------------
    // Entities
    // ------------------------------------------------------------------
    class Entity
    /*
    * The base class of anything that has a time and space in the world.
    * Use StaticEntity or DynamicEntity instead of this directly.
    */
    {
        friend class World;

    public:
        Entity(World* ownerWorld, const transformf& spawnTransform)
            : transform(spawnTransform), OwnerWorld(ownerWorld)
        {
        }

        virtual ~Entity() = default;

        World* GetWorld() const { return OwnerWorld; }

        const transformf& GetTransform() const { return transform; }
        transformf& GetTransform() { return transform; }

        virtual void UpdateTransform() {}
        virtual void UpdateRender() {}

        // Local-space collision box used by World::LineTrace (before the entity's transform
        // is applied). Defaults to a unit cube; set from the mesh bounds after spawning.
        vec3f BoundsMin{ -0.5f, -0.5f, -0.5f };
        vec3f BoundsMax{ 0.5f, 0.5f, 0.5f };

    protected:
        transformf transform;
        World* OwnerWorld = nullptr;
        std::size_t worldIndex = static_cast<std::size_t>(-1);
    };

    class StaticEntity : public Entity
    {
    public:
        StaticEntity(World* ownerWorld, const transformf& spawnTransform, StaticMesh* mesh = nullptr)
            : Entity(ownerWorld, spawnTransform), Mesh(mesh)
        {
            CachedMatrix = toMat4(spawnTransform);
        }

        StaticMesh* Mesh = nullptr;
        mat4f CachedMatrix = scaling(2.0f);

    private:
        void UpdateTransform() override {}
        void UpdateRender() override {}
    };

    class DynamicEntity : public Entity
    {
    public:
        DynamicEntity(World* ownerWorld, const transformf& spawnTransform, SkinnedMesh* mesh = nullptr)
            : Entity(ownerWorld, spawnTransform), Mesh(mesh)
        {
        }

        SkinnedMesh* Mesh = nullptr;

    private:
        void UpdateTransform() override {}
        void UpdateRender() override {}
    };

    // ------------------------------------------------------------------
    // World
    // ------------------------------------------------------------------
    class World
    {
    public:
        struct Slot {
            std::atomic<Entity*> entity{ nullptr };
            std::atomic<bool> isOccupied{ false };
        };

        explicit World(const WorldConfig& config = {})
            : maxDynamicCapacity(config.maxDynamic), maxStaticCapacity(config.maxStatic)
        {
            slots = std::make_unique<Slot[]>(maxDynamicCapacity);
            staticEntities.reserve(maxStaticCapacity);
        }

        explicit World(worldtypes type) : World(GetWorldConfig(type)) {}

        // Entities hold a raw World*, so a World must never move or be copied.
        // Store worlds as std::unique_ptr<World>.
        World(const World&) = delete;
        World& operator=(const World&) = delete;
        World(World&&) = delete;
        World& operator=(World&&) = delete;

        virtual ~World()
        {
            if (slots) {
                for (std::size_t i = 0; i < maxDynamicCapacity; ++i) {
                    if (slots[i].isOccupied.load(std::memory_order_acquire)) {
                        delete slots[i].entity.load(std::memory_order_relaxed);
                    }
                }
            }
            for (StaticEntity* ent : staticEntities) {
                delete ent;
            }
        }

        __forceinline bool IsDormant() const { return dormant; }
        __forceinline void SetDormant(bool value) { dormant = value; }

        template <typename T, typename... Args>
        T* CreateEntity(const transformf& spawnTransform, Args&&... args)
        {
            static_assert(std::is_base_of_v<DynamicEntity, T>, "T must derive from DynamicEntity");

            for (std::size_t i = 0; i < maxDynamicCapacity; ++i) {
                bool expected = false;
                if (slots[i].isOccupied.compare_exchange_strong(expected, true,
                        std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    T* newEntity = new T(this, spawnTransform, std::forward<Args>(args)...);
                    newEntity->worldIndex = i;
                    slots[i].entity.store(newEntity, std::memory_order_release);
                    return newEntity;
                }
            }
            return nullptr;
        }

        template <typename T, typename... Args>
        T* CreateStaticEntity(const transformf& spawnTransform, Args&&... args)
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

            const std::size_t idx = entity->worldIndex;
            if (idx >= maxDynamicCapacity) return;

            Entity* expectedEntity = entity;
            if (slots[idx].entity.compare_exchange_strong(expectedEntity, nullptr,
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                slots[idx].isOccupied.store(false, std::memory_order_release);
                delete entity;   // NOTE: unsafe if another thread is mid-LineTrace/UpdateWorld on it
            }
        }

        void UpdateWorld()
        {
            if (dormant) return;

            for (std::size_t i = 0; i < maxDynamicCapacity; ++i) {
                if (slots[i].isOccupied.load(std::memory_order_acquire)) {
                    Entity* ent = slots[i].entity.load(std::memory_order_acquire);
                    if (ent) {
                        ent->UpdateTransform();
                        ent->UpdateRender();
                    }
                }
            }
        }

        // Segment from start to end against every entity's oriented bounding box.
        // Returns true and fills outHit with the closest hit. A box that already
        // contains `start` is ignored (no entry point).
        bool LineTrace(const vec3f& start, const vec3f& end, TraceHit& outHit,
                       const TraceParams& params = {}) const
        {
            const vec3f delta = end - start;
            const float segLength = length(delta);
            if (segLength < 1e-6f) return false;

            float bestT = 2.0f;                 // segment fraction, 0..1
            vec3f bestNormal{};
            Entity* bestEntity = nullptr;

            auto consider = [&](Entity* ent, const mat4f& model) {
                if (!ent || ent == params.ignore) return;
                float t; vec3f n;
                if (TraceBox(model, ent->BoundsMin, ent->BoundsMax, start, delta, t, n) && t < bestT) {
                    bestT = t; bestNormal = n; bestEntity = ent;
                }
            };

            if (params.traceDynamic) {
                for (std::size_t i = 0; i < maxDynamicCapacity; ++i) {
                    if (!slots[i].isOccupied.load(std::memory_order_acquire)) continue;
                    Entity* ent = slots[i].entity.load(std::memory_order_acquire);
                    if (ent) consider(ent, toMat4(ent->GetTransform()));
                }
            }

            if (params.traceStatic) {
                for (StaticEntity* ent : staticEntities) {
                    if (ent) consider(ent, ent->CachedMatrix);
                }
            }

            if (!bestEntity) return false;

            outHit.entity = bestEntity;
            outHit.distance = bestT * segLength;
            outHit.position = start + delta * bestT;
            outHit.normal = bestNormal;
            return true;
        }

        std::unique_ptr<Slot[]> slots;
        std::vector<StaticEntity*> staticEntities;
        std::size_t maxDynamicCapacity = 0;
        std::size_t maxStaticCapacity = 0;
        std::uint8_t depth = 0;

    private:
        bool dormant = true;

        // Slab test in the box's local space. `dir` is the un-normalized segment vector, so
        // outT is a 0..1 fraction of the segment. Handles rotation and non-uniform scale.
        static bool TraceBox(const mat4f& model, const vec3f& bmin, const vec3f& bmax,
                             const vec3f& start, const vec3f& dir,
                             float& outT, vec3f& outNormal)
        {
            // Entity matrices are affine (T*R*S), so only the 3x3 part can be degenerate.
            if (std::abs(determinant(toMat3(model))) < 1e-12f) return false;   // zero scale

            const mat4f invModel = inverseAffine(model);   // cheaper than the general 4x4 inverse
            const vec3f o = transformPoint(invModel, start);    // w = 1
            const vec3f d = transformVector(invModel, dir);     // w = 0

            float tMin = 0.0f, tMax = 1.0f;
            int axis = -1;
            float sign = 0.0f;

            for (int i = 0; i < 3; ++i) {
                if (std::abs(d[i]) < 1e-8f) {                       // parallel to this slab
                    if (o[i] < bmin[i] || o[i] > bmax[i]) return false;
                    continue;
                }
                const float invD = 1.0f / d[i];
                float t0 = (bmin[i] - o[i]) * invD;
                float t1 = (bmax[i] - o[i]) * invD;
                float s = -1.0f;                                    // entering through the min face
                if (t0 > t1) { std::swap(t0, t1); s = 1.0f; }       // entering through the max face
                if (t0 > tMin) { tMin = t0; axis = i; sign = s; }
                if (t1 < tMax) tMax = t1;
                if (tMin > tMax) return false;
            }

            if (axis < 0) return false;                             // start is inside the box

            vec3f localN{};
            localN[static_cast<std::size_t>(axis)] = sign;
            // normals transform by transpose(inverse(upper 3x3)); invModel is already the inverse
            outNormal = normalize(transpose(toMat3(invModel)) * localN);
            outT = tMin;
            return true;
        }
    };

} // namespace R7::WorldFramework
