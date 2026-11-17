
#pragma once

// ==========================================================
// R7DataTypes - frequent-use data types with math helpers
// Copyright (c) 2026 RedSeven (Licensed under the MIT License)
//
// boneless templated types
// ==========================================================


/*
* R7 DataTypes library
* includes math functions
* 
* includes chain inherited vectors for data slicing.
* quat,transforms
* all templated, enjoy.
* 
* if you find any bugs shoot me a message on github at 
* 
* RedSeven:Red7Seven7
*/
#pragma region includes
#include <type_traits>
#include <concepts>
#pragma endregion
#pragma region concepts
template<typename t>
concept scalar = std::is_arithmetic_v<t>;
#pragma endregion
#pragma region macros
#define SCALT template<scalar t>
#define SCALTND SCALT [[nodiscard]]
#define REGT template<typename t>
#define REGTND REGT [[nodiscard]]
#define FLTT template <std::floating_point t>
#define FLTTND FLTT [[nodiscard]]
#define FLTTICX FLTT inline constexpr
#define FLTTCX FLTT constexpr
#define NDFRCX [[nodiscard]] friend constexpr
#define NDCX [[nodiscoard]] constexpr
#pragma endregion
#pragma region constants
/* Usage: pi<float>, tau<double>, epsilon<float>, ... */
FLTTICX t pi = std::numbers::pi_v<t>;
FLTTICX t tau = t(2) * std::numbers::pi_v<t>;
FLTTICX t halfPi = std::numbers::pi_v<t> / t(2);
FLTTICX t quarterPi = std::numbers::pi_v<t> / t(4);
FLTTICX t invPi = std::numbers::inv_pi_v<t>;
FLTTICX t euler = std::numbers::e_v<t>;
FLTTICX t sqrt2 = std::numbers::sqrt2_v<t>;
FLTTICX t sqrt3 = std::numbers::sqrt3_v<t>;
FLTTICX t ln2 = std::numbers::ln2_v<t>;
FLTTICX t ln10 = std::numbers::ln10_v<t>;
FLTTICX t phi = std::numbers::phi_v<t>;   // golden ratio

/* machine epsilon (smallest x where 1 + x != 1), and the "close enough" tolerance used by approxEqual */
FLTTICX t machineEpsilon = std::numeric_limits<t>::epsilon();
FLTTICX t epsilon = t(1e-5);

FLTTICX t infinity = std::numeric_limits<t>::infinity();
FLTTICX t maxValue = std::numeric_limits<t>::max();
FLTTICX t minValue = std::numeric_limits<t>::lowest(); // most negative; min() is the smallest positive

/* angle conversion factors */
FLTTICX t degToRad = std::numbers::pi_v<t> / t(180);
FLTTICX t radToDeg = t(180) / std::numbers::pi_v<t>;

FLTTCX t toRadians(t degrees) noexcept { return degrees * degToRad<t>; }
FLTTCX t toDegrees(t radians) noexcept { return radians * radToDeg<t>; }

/* float / double shortcuts */
FLTTICX float  pif = pi<float>;
FLTTICX double pid = pi<double>;
FLTTICX float  taof = tau<float>;
FLTTICX double taod = tau<double>;
#pragma endregion
#pragma region Matrices
SCALT struct matrix2x2 { t m[2][2]; }; typedef matrix2x2<float> mat2f;
SCALT struct matrix3x3 { t m[3][3]; }; typedef matrix3x3<float> mat3f;
SCALT struct matrix4x4 { t m[4][4]; }; typedef matrix4x4<float> mat4f;
#pragma endregion
#pragma region vector types
#pragma region vector2
SCALT struct vec2 {
    t x{};
    t y{};
    /* unary */
    NDCX vec2 operator+() const noexcept { return *this; }
    NDCX vec2 operator-() const noexcept { return { -x, -y }; }
    /* compound */
    constexpr vec2& operator+=(const vec2& r) noexcept { x += r.x; y += r.y; return *this; }
    constexpr vec2& operator-=(const vec2& r) noexcept { x -= r.x; y -= r.y; return *this; }
    constexpr vec2& operator*=(t s) noexcept { x *= s; y *= s; return *this; }
    constexpr vec2& operator/=(t s) noexcept { x /= s; y /= s; return *this; }
    /* binary friends */
    NDFRCX vec2 operator+(vec2 l, const vec2& r) noexcept { return l += r; }
    NDFRCX vec2 operator-(vec2 l, const vec2& r) noexcept { return l -= r; }
    NDFRCX vec2 operator*(vec2 v, t s) noexcept { return v *= s; }
    NDFRCX vec2 operator*(t s, vec2 v) noexcept { return v *= s; }
    NDFRCX vec2 operator/(vec2 v, t s) noexcept { return v /= s; }
    /* comparrison */
    NDFRCX bool operator==(const vec2&, const vec2&) = default;
    /* indexed access */
    NDCX t& operator[](std::size_t i) noexcept { return i == 0 ? x : y; }
    NDCX const t& operator[](std::size_t i) const noexcept { return i == 0 ? x : y; }
};
SCALTND constexpr t dot(const vec2<t>& a, const vec2<t>& b) noexcept { return a.x * b.x + a.y * b.y; }
SCALTND constexpr t cross(const vec2<t>& a, const vec2<t>& b) noexcept { return a.x * b.y - a.y * b.x; }
SCALTND constexpr t lengthSquared(const vec2<t>& v) noexcept { return dot(v, v); }
FLTTND t length(const vec2<t>& v) noexcept { return std::sqrt(lengthSquared(v)); }
FLTTND vec2<t> normalize(const vec2<t>& v) noexcept {
    const t len = length(v);
    return len > t(0) ? v / len : vec2<t>{};
}
FLTTND constexpr bool approxEqual(const vec2<t>& a, const vec2<t>& b, t eps = t(1e-5)) noexcept {
    const t dx = a.x - b.x, dy = a.y - b.y;
    return (dx < t(0) ? -dx : dx) <= eps && (dy < t(0) ? -dy : dy) <= eps;
}
#pragma endregion
#pragma region vector3
SCALT struct vec3 {
    t x{};
    t y{};
    t z{};

    /*unary*/
    [[nodiscard]] constexpr vec3 operator+() const noexcept { return *this; }
    [[nodiscard]] constexpr vec3 operator-() const noexcept { return { -x, -y, -z }; }

    /*compound assignment*/
    constexpr vec3& operator+=(const vec3& r) noexcept { x += r.x; y += r.y; z += r.z; return *this; }
    constexpr vec3& operator-=(const vec3& r) noexcept { x -= r.x; y -= r.y; z -= r.z; return *this; }
    constexpr vec3& operator*=(t s) noexcept { x *= s; y *= s; z *= s; return *this; }
    constexpr vec3& operator/=(t s) noexcept { x /= s; y /= s; z /= s; return *this; }

    /*binary friends*/
    NDFRCX vec3 operator+(vec3 l, const vec3& r) noexcept { return l += r; }
    NDFRCX vec3 operator-(vec3 l, const vec3& r) noexcept { return l -= r; }
    NDFRCX vec3 operator*(vec3 v, t s) noexcept { return v *= s; }
    NDFRCX vec3 operator*(t s, vec3 v) noexcept { return v *= s; }
    NDFRCX vec3 operator/(vec3 v, t s) noexcept { return v /= s; }

    /*comparison*/
    NDFRCX bool operator==(const vec3&, const vec3&) = default;

    /*indexed access, 0=x,1=y,2=z (anything else returns z)*/
    NDFRCX t& operator[](std::size_t i) noexcept { return i == 0 ? x : (i == 1 ? y : z); }
    NDFRCX const t& operator[](std::size_t i) const noexcept { return i == 0 ? x : (i == 1 ? y : z); }
};
SCALTND constexpr t dot(const vec3<t>& a, const vec3<t>& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
SCALTND constexpr vec3<t> cross(const vec3<t>& a, const vec3<t>& b) noexcept {
    return { a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}
SCALTND constexpr t lengthSquared(const vec3<t>& v) noexcept { return dot(v, v); }
FLTTND t length(const vec3<t>& v) noexcept { return std::sqrt(lengthSquared(v)); }
FLTTND vec3<t> normalize(const vec3<t>& v) noexcept {
    const t len = length(v);
    return len > t(0) ? v / len : vec3<t>{};
}
FLTTND constexpr bool approxEqual(const vec3<t>& a, const vec3<t>& b, t eps = t(1e-5)) noexcept {
    const t dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return (dx < t(0) ? -dx : dx) <= eps
        && (dy < t(0) ? -dy : dy) <= eps
        && (dz < t(0) ? -dz : dz) <= eps;
}
#pragma endregion
#pragma region vector4
SCALT struct vec4 {
    t x{};
    t y{};
    t z{};
    t w{};

    /*unary*/
    NDCX vec4 operator+() const noexcept { return *this; }
    NDCX vec4 operator-() const noexcept { return { -x, -y, -z, -w }; }

    /*compound assignment*/
    constexpr vec4& operator+=(const vec4& r) noexcept { x += r.x; y += r.y; z += r.z; w += r.w; return *this; }
    constexpr vec4& operator-=(const vec4& r) noexcept { x -= r.x; y -= r.y; z -= r.z; w -= r.w; return *this; }
    constexpr vec4& operator*=(t s) noexcept { x *= s; y *= s; z *= s; w *= s; return *this; }
    constexpr vec4& operator/=(t s) noexcept { x /= s; y /= s; z /= s; w /= s; return *this; }

    /*binary friends*/
    NDFRCX vec4 operator+(vec4 l, const vec4& r) noexcept { return l += r; }
    NDFRCX vec4 operator-(vec4 l, const vec4& r) noexcept { return l -= r; }
    NDFRCX vec4 operator*(vec4 v, t s) noexcept { return v *= s; }
    NDFRCX vec4 operator*(t s, vec4 v) noexcept { return v *= s; }
    NDFRCX vec4 operator/(vec4 v, t s) noexcept { return v /= s; }

    /*comparison*/
    NDFRCX bool operator==(const vec4&, const vec4&) = default;

    /*indexed access, 0=x,1=y,2=z,3=w (anything else returns w)*/
    NDCX t& operator[](std::size_t i) noexcept {
        return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
    }
    NDCX const t& operator[](std::size_t i) const noexcept {
        return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
    }
};
SCALTND constexpr t dot(const vec4<t>& a, const vec4<t>& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
SCALTND constexpr t lengthSquared(const vec4<t>& v) noexcept { return dot(v, v); }
FLTTND t length(const vec4<t>& v) noexcept { return std::sqrt(lengthSquared(v)); }
FLTTND vec4<t> normalize(const vec4<t>& v) noexcept {
    const t len = length(v);
    return len > t(0) ? v / len : vec4<t>{};
}
FLTTND constexpr bool approxEqual(const vec4<t>& a, const vec4<t>& b, t eps = t(1e-5)) noexcept {
    const t dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z, dw = a.w - b.w;
    return (dx < t(0) ? -dx : dx) <= eps
        && (dy < t(0) ? -dy : dy) <= eps
        && (dz < t(0) ? -dz : dz) <= eps
        && (dw < t(0) ? -dw : dw) <= eps;
}
#pragma endregion
#pragma endregion
#pragma region quat
FLTT struct quat {
    t x{};
    t y{};
    t z{};
    t w{ t(1) };

    /*factories*/
    [[nodiscard]] static constexpr quat identity() noexcept { return { t(0), t(0), t(0), t(1) }; }

    /* axis need not be normalized; angle in radians */
    [[nodiscard]] static quat fromAxisAngle(const vec3<t>& axis, t angle) noexcept {
        const vec3<t> n = normalize(axis);
        const t half = angle * t(0.5);
        const t s = std::sin(half);
        return { n.x * s, n.y * s, n.z * s, std::cos(half) };
    }

    /*unary*/
    NDCX quat operator+() const noexcept { return *this; }
    NDCX quat operator-() const noexcept { return { -x, -y, -z, -w }; }

    /*compound assignment*/
    constexpr quat& operator+=(const quat& r) noexcept { x += r.x; y += r.y; z += r.z; w += r.w; return *this; }
    constexpr quat& operator-=(const quat& r) noexcept { x -= r.x; y -= r.y; z -= r.z; w -= r.w; return *this; }
    constexpr quat& operator*=(t s) noexcept { x *= s; y *= s; z *= s; w *= s; return *this; }
    constexpr quat& operator/=(t s) noexcept { x /= s; y /= s; z /= s; w /= s; return *this; }
    constexpr quat& operator*=(const quat& r) noexcept { *this = *this * r; return *this; }

    /*binary friends*/
    NDFRCX quat operator+(quat l, const quat& r) noexcept { return l += r; }
    NDFRCX quat operator-(quat l, const quat& r) noexcept { return l -= r; }
    NDFRCX quat operator*(quat q, t s) noexcept { return q *= s; }
    NDFRCX quat operator*(t s, quat q) noexcept { return q *= s; }
    NDFRCX quat operator/(quat q, t s) noexcept { return q /= s; }

    /* Hamilton product: (l * r) applies r first, then l */
    NDFRCX quat operator*(const quat& l, const quat& r) noexcept {
        return {
            l.w * r.x + l.x * r.w + l.y * r.z - l.z * r.y,
            l.w * r.y - l.x * r.z + l.y * r.w + l.z * r.x,
            l.w * r.z + l.x * r.y - l.y * r.x + l.z * r.w,
            l.w * r.w - l.x * r.x - l.y * r.y - l.z * r.z
        };
    }

    /*comparison*/
    NDFRCX bool operator==(const quat&, const quat&) = default;

    /*indexed access, 0=x,1=y,2=z,3=w (anything else returns w)*/
    NDCX t& operator[](std::size_t i) noexcept {
        return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
    }
    NDCX const t& operator[](std::size_t i) const noexcept {
        return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
    }
};
FLTTND constexpr t dot(const quat<t>& a, const quat<t>& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
FLTTND constexpr t lengthSquared(const quat<t>& q) noexcept { return dot(q, q); }
FLTTND t length(const quat<t>& q) noexcept { return std::sqrt(lengthSquared(q)); }
FLTTND quat<t> normalize(const quat<t>& q) noexcept {
    const t len = length(q);
    return len > t(0) ? q / len : quat<t>::identity();
}
FLTTND constexpr quat<t> conjugate(const quat<t>& q) noexcept { return { -q.x, -q.y, -q.z, q.w }; }
/* for a unit quaternion this equals conjugate(q), which is cheaper */
FLTTND constexpr quat<t> inverse(const quat<t>& q) noexcept {
    const t lsq = lengthSquared(q);
    return lsq > t(0) ? conjugate(q) / lsq : quat<t>::identity();
}
/* rotates v by unit quaternion q */
FLTTND constexpr vec3<t> rotate(const quat<t>& q, const vec3<t>& v) noexcept {
    const vec3<t> u{ q.x, q.y, q.z };
    const vec3<t> uv = cross(u, v);
    const vec3<t> uuv = cross(u, uv);
    return v + (uv * q.w + uuv) * t(2);
}
/* spherical interpolation along the shortest arc, t in [0,1] */
FLTTND quat<t> slerp(const quat<t>& a, quat<t> b, t f) noexcept {
    t d = dot(a, b);
    if (d < t(0)) { b = -b; d = -d; }              // take the short way around
    if (d > t(0.9995)) return normalize(a + (b - a) * f); // nearly parallel: nlerp
    const t theta = std::acos(d);
    const t sinTheta = std::sin(theta);
    const t wa = std::sin((t(1) - f) * theta) / sinTheta;
    const t wb = std::sin(f * theta) / sinTheta;
    return a * wa + b * wb;
}
FLTTND constexpr bool approxEqual(const quat<t>& a, const quat<t>& b, t eps = t(1e-5)) noexcept {
    const t dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z, dw = a.w - b.w;
    return (dx < t(0) ? -dx : dx) <= eps
        && (dy < t(0) ? -dy : dy) <= eps
        && (dz < t(0) ? -dz : dz) <= eps
        && (dw < t(0) ? -dw : dw) <= eps;
}
#pragma endregion
#pragma region transform
FLTT struct transform {
    vec3<t> position{};
    quat<t> rotation{};                    // defaults to identity
    vec3<t> scale{ t(1), t(1), t(1) };

    [[nodiscard]] static constexpr transform identity() noexcept { return {}; }

    /* point: scaled, rotated, then translated */
    NDCX vec3<t> transformPoint(const vec3<t>& p) const noexcept {
        const vec3<t> s{ p.x * scale.x, p.y * scale.y, p.z * scale.z };
        return rotate(rotation, s) + position;
    }

    /* direction: scaled and rotated, translation ignored */
    NDCX vec3<t> transformVector(const vec3<t>& v) const noexcept {
        const vec3<t> s{ v.x * scale.x, v.y * scale.y, v.z * scale.z };
        return rotate(rotation, s);
    }

    /* parent * child: child is applied first, then parent.
       Exact for uniform scale; with non-uniform scale and rotated parents, shear is dropped. */
    NDFRCX transform operator*(const transform& p, const transform& c) noexcept {
        return {
            p.transformPoint(c.position),
            p.rotation * c.rotation,
            { p.scale.x * c.scale.x, p.scale.y * c.scale.y, p.scale.z * c.scale.z }
        };
    }

    NDFRCX bool operator==(const transform&, const transform&) = default;
};

/* Exact only for uniform scale. A non-uniformly scaled, rotated transform has an inverse
   that can't be written as position/rotation/scale; use the matrix form for that. */
FLTTND constexpr transform<t> inverse(const transform<t>& tr) noexcept {
    const quat<t> invRot = conjugate(tr.rotation);
    const vec3<t> invScale{ t(1) / tr.scale.x, t(1) / tr.scale.y, t(1) / tr.scale.z };
    const vec3<t> scaledPos{ tr.position.x * invScale.x, tr.position.y * invScale.y, tr.position.z * invScale.z };
    return { -rotate(invRot, scaledPos), invRot, invScale };
}

/* Column-vector convention, m[row][col]: M = T * R * S */
FLTTND constexpr matrix4x4<t> toMat4(const transform<t>& tr) noexcept {
    const t x = tr.rotation.x, y = tr.rotation.y, z = tr.rotation.z, w = tr.rotation.w;
    const t r[3][3] = {
        { t(1) - t(2) * (y * y + z * z), t(2) * (x * y - z * w),        t(2) * (x * z + y * w) },
        { t(2) * (x * y + z * w),        t(1) - t(2) * (x * x + z * z), t(2) * (y * z - x * w) },
        { t(2) * (x * z - y * w),        t(2) * (y * z + x * w),        t(1) - t(2) * (x * x + y * y) }
    };
    const t s[3] = { tr.scale.x, tr.scale.y, tr.scale.z };
    const t p[3] = { tr.position.x, tr.position.y, tr.position.z };

    matrix4x4<t> out{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) out.m[i][j] = r[i][j] * s[j]; // scale each column of R
        out.m[i][3] = p[i];
    }
    out.m[3][3] = t(1);
    return out;
}

typedef transform<float> transformf;
#pragma endregion
