
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
template<typename t>concept scalar = std::is_arithmetic_v<t>;
/* true for vec2<T>, vec3<T>, vec4<T> */
template<typename v> concept vector_type = requires { vecTraits<v>::dim; };

#pragma endregion
#pragma region macros
#define SCALT template<scalar t>
#define SCALTND SCALT [[nodiscard]]
#define SCALTNDCX SCALTND constexpr
#define REGT template<typename t>
#define REGTND REGT [[nodiscard]]
#define FLTT template <std::floating_point t>
#define FLTTND FLTT [[nodiscard]]
#define FLTTNDCX FLTTND constexpr
#define FLTTICX FLTT inline constexpr
#define FLTTCX FLTT constexpr
#define NDFRCX [[nodiscard]] friend constexpr
#define NDCX [[nodiscoard]] constexpr
#define TV template<vector_type v>
#define TVCX TV constexpr
#define TVND [[nodiscard]]
#define TVNDCX TV constexpr [[nodiscard]]
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
#pragma region ExtraMath
#pragma region floor
SCALTND constexpr t floor(t x) noexcept {
    if constexpr (std::floating_point<t>) {
        constexpr t limit = t(1) / std::numeric_limits<t>::epsilon();
        if (!(x > -limit && x < limit)) return x;
        if (x == t(0)) return x;                        // preserves -0.0
        const t i = static_cast<t>(static_cast<long long>(x)); // truncates toward zero
        return i > x ? i - t(1) : i;                    // fix up negative non-integers
    }
    else {
        return x;
    }
}
/* component-wise  */
SCALTNDCX vec2<t> floor(const vec2<t>& v) noexcept { return { floor(v.x), floor(v.y) }; }
SCALTNDCX vec3<t> floor(const vec3<t>& v) noexcept { return { floor(v.x), floor(v.y), floor(v.z) }; }
SCALTNDCX vec4<t> floor(const vec4<t>& v) noexcept { return { floor(v.x), floor(v.y), floor(v.z), floor(v.w) }; }
#pragma endregion
#pragma region fastFloor
/*NO PROTECTION FLOOR*/
template<std::floating_point t, std::signed_integral R = int>
NDCX R floorToInt(t x) noexcept {
    const R i = static_cast<R>(x);                  // truncates toward zero
    return i - static_cast<R>(x < static_cast<t>(i)); // subtract 1 for negative non-integers
}
FLTTNDCX t fastFloor(t x) noexcept {return static_cast<t>(floorToInt<t, int>(x));}
/* component-wise */
FLTTNDCX vec2<t> fastFloor(const vec2<t>& v) noexcept { return { fastFloor(v.x), fastFloor(v.y) }; }
FLTTNDCX vec3<t> fastFloor(const vec3<t>& v) noexcept { return { fastFloor(v.x), fastFloor(v.y), fastFloor(v.z) }; }
FLTTNDCX vec4<t> fastFloor(const vec4<t>& v) noexcept { return { fastFloor(v.x), fastFloor(v.y), fastFloor(v.z), fastFloor(v.w) }; }
#pragma endregion
#pragma region scalar
#pragma region basic
SCALTNDCX t sqr(t x) noexcept { return x * x; }
SCALTNDCX t abs(t x) noexcept {
    if constexpr (std::is_unsigned_v<t>) return x;
    else return x < t(0) ? -x : x;
}

SCALTNDCX t min(t a, t b) noexcept { return b < a ? b : a; }
SCALTNDCX t max(t a, t b) noexcept { return a < b ? b : a; }

SCALTNDCX t clamp(t x, t lo, t hi) noexcept { return x < lo ? lo : (hi < x ? hi : x); }
FLTTNDCX t saturate(t x) noexcept { return clamp(x, t(0), t(1)); }

SCALTNDCX t sign(t x) noexcept {
    return static_cast<t>(static_cast<t>(t(0) < x) - static_cast<t>(x < t(0)));
}
FLTTND t rsqrt(t x) noexcept { return t(1) / std::sqrt(x); }
#pragma endregion
#pragma region interpolation
/* exact at f = 0 and f = 1; f is not clamped, so values outside [0,1] extrapolate */
FLTTNDCX t lerp(t a, t b, t f) noexcept { return (t(1) - f) * a + f * b; }
/* where x sits between a and b (0 at a, 1 at b); returns 0 if a == b */
FLTTNDCX t inverseLerp(t a, t b, t x) noexcept { return b != a ? (x - a) / (b - a) : t(0); }
FLTTNDCX t remap(t x, t inLo, t inHi, t outLo, t outHi) noexcept {
    return lerp(outLo, outHi, inverseLerp(inLo, inHi, x));
}
/* 0 if x < edge, otherwise 1 */
FLTTNDCX t step(t edge, t x) noexcept { return x < edge ? t(0) : t(1); }
/* smooth Hermite ramp from 0 at e0 to 1 at e1 */
FLTTNDCX t smoothstep(t e0, t e1, t x) noexcept {
    if (e0 == e1) return step(e0, x);
    const t u = saturate((x - e0) / (e1 - e0));
    return u * u * (t(3) - t(2) * u);
}
#pragma endregion
#pragma region rounding
/* all of these return integers unchanged; NaN and infinity pass through */
SCALTNDCX t ceil(t x) noexcept {
    if constexpr (std::floating_point<t>) return -floor(-x);
    else return x;
}

/* rounds toward zero */
SCALTNDCX t trunc(t x) noexcept {
    if constexpr (std::floating_point<t>) return x < t(0) ? ceil(x) : floor(x);
    else return x;
}

/* rounds half away from zero, like std::round.
   Computed from the fractional part, so 0.49999999999999994 rounds correctly to 0. */
SCALTNDCX t round(t x) noexcept 
{
    if constexpr (std::floating_point<t>) {
        const t i = trunc(x);
        const t diff = abs(x - i);
        return diff >= t(0.5) ? i + (x < t(0) ? t(-1) : t(1)) : i;
    }
    else {
        return x;
    }
}

/* x - floor(x), always in [0,1) (GLSL-style: fract(-0.25) = 0.75) */
FLTTND constexpr t fract(t x) noexcept { return x - floor(x); }
#pragma endregion
#pragma region modulo and wrap
/* Floored modulo: the result takes the sign of y, so mod(-1, 5) = 4 (unlike C++'s % and fmod).
   Integer y must not be 0. */
SCALTNDCX t mod(t x, t y) noexcept {
    if constexpr (std::floating_point<t>) {
        return x - y * floor(x / y);
    }
    else {
        t r = x % y;
        if constexpr (!std::is_unsigned_v<t>) {
            if (r != t(0) && ((r < t(0)) != (y < t(0)))) r += y;
        }
        return r;
    }
}
/* wraps x into [lo, hi), e.g. angles or tile coordinates. Use signed or floating types
   (unsigned underflows when x < lo). Requires hi > lo. */
SCALTNDCX t wrap(t x, t lo, t hi) noexcept { return lo + mod(x - lo, hi - lo); }
#pragma endregion
#pragma region powers of two
template<std::integral t>
NDCX bool isPowerOfTwo(t x) noexcept { return x > t(0) && (x & (x - t(1))) == t(0); }

/* smallest power of two >= x (returns 1 for 0 and 1). Returns 0 if the result would overflow t. */
template<std::unsigned_integral t>
NDCX t nextPowerOfTwo(t x) noexcept {
    if (x <= t(1)) return t(1);
    --x;
    for (unsigned s = 1; s < static_cast<unsigned>(std::numeric_limits<t>::digits); s <<= 1) x |= static_cast<t>(x >> s);
    return static_cast<t>(x + t(1));
}
#pragma endregion
#pragma endregion
#pragma region matrixOps
#pragma region matrix traits
template<typename mt> struct matTraits {};

template<scalar t> struct matTraits<matrix2x2<t>> {
    using scalar_type = t; using vec_type = vec2<t>; static constexpr std::size_t dim = 2;
};
template<scalar t> struct matTraits<matrix3x3<t>> {
    using scalar_type = t; using vec_type = vec3<t>; static constexpr std::size_t dim = 3;
};
template<scalar t> struct matTraits<matrix4x4<t>> {
    using scalar_type = t; using vec_type = vec4<t>; static constexpr std::size_t dim = 4;
};

/* true for matrix2x2<T>, matrix3x3<T>, matrix4x4<T> */
template<typename mt> concept matrix_type = requires { matTraits<mt>::dim; };

template<matrix_type mt> using matScalarOf = typename matTraits<mt>::scalar_type;
template<matrix_type mt> using matVecOf = typename matTraits<mt>::vec_type;
template<matrix_type mt> inline constexpr std::size_t matDimOf = matTraits<mt>::dim;

#define TM template<matrix_type mt>
#define TMCX TM constexpr
#define TMNDCX TM [[nodiscard]] constexpr
#define TMFLT template<matrix_type mt> requires std::floating_point<matScalarOf<mt>>

typedef matrix2x2<double> mat2d;
typedef matrix3x3<double> mat3d;
typedef matrix4x4<double> mat4d;
#pragma endregion
#pragma region matrix construction and access
/* usage: matIdentity<mat4f>() */
TMNDCX mt matIdentity() noexcept {
    mt r{};
    for (std::size_t i = 0; i < matDimOf<mt>; ++i) r.m[i][i] = matScalarOf<mt>(1);
    return r;
}

/* build from basis vectors (each vector becomes a column) */
SCALTNDCX matrix2x2<t> fromColumns(const vec2<t>& c0, const vec2<t>& c1) noexcept {
    return {{ { c0.x, c1.x },
              { c0.y, c1.y } }};
}
SCALTNDCX matrix3x3<t> fromColumns(const vec3<t>& c0, const vec3<t>& c1, const vec3<t>& c2) noexcept {
    return {{ { c0.x, c1.x, c2.x },
              { c0.y, c1.y, c2.y },
              { c0.z, c1.z, c2.z } }};
}
SCALTNDCX matrix4x4<t> fromColumns(const vec4<t>& c0, const vec4<t>& c1, const vec4<t>& c2, const vec4<t>& c3) noexcept {
    return {{ { c0.x, c1.x, c2.x, c3.x },
              { c0.y, c1.y, c2.y, c3.y },
              { c0.z, c1.z, c2.z, c3.z },
              { c0.w, c1.w, c2.w, c3.w } }};
}

/* row / column i as a vector (i must be < dimension) */
TMNDCX matVecOf<mt> row(const mt& a, std::size_t i) noexcept {
    if constexpr (matDimOf<mt> == 2) return { a.m[i][0], a.m[i][1] };
    else if constexpr (matDimOf<mt> == 3) return { a.m[i][0], a.m[i][1], a.m[i][2] };
    else return { a.m[i][0], a.m[i][1], a.m[i][2], a.m[i][3] };
}
TMNDCX matVecOf<mt> column(const mt& a, std::size_t i) noexcept {
    if constexpr (matDimOf<mt> == 2) return { a.m[0][i], a.m[1][i] };
    else if constexpr (matDimOf<mt> == 3) return { a.m[0][i], a.m[1][i], a.m[2][i] };
    else return { a.m[0][i], a.m[1][i], a.m[2][i], a.m[3][i] };
}
#pragma endregion
#pragma region matrix arithmetic
TMNDCX mt operator+(const mt& a) noexcept { return a; }
TMNDCX mt operator-(const mt& a) noexcept {
    mt r{};
    for (std::size_t i = 0; i < matDimOf<mt>; ++i)
        for (std::size_t j = 0; j < matDimOf<mt>; ++j) r.m[i][j] = -a.m[i][j];
    return r;
}

/* compound */
TMCX mt& operator+=(mt& a, const mt& b) noexcept {
    for (std::size_t i = 0; i < matDimOf<mt>; ++i)
        for (std::size_t j = 0; j < matDimOf<mt>; ++j) a.m[i][j] += b.m[i][j];
    return a;
}
TMCX mt& operator-=(mt& a, const mt& b) noexcept {
    for (std::size_t i = 0; i < matDimOf<mt>; ++i)
        for (std::size_t j = 0; j < matDimOf<mt>; ++j) a.m[i][j] -= b.m[i][j];
    return a;
}
TMCX mt& operator*=(mt& a, matScalarOf<mt> s) noexcept {
    for (std::size_t i = 0; i < matDimOf<mt>; ++i)
        for (std::size_t j = 0; j < matDimOf<mt>; ++j) a.m[i][j] *= s;
    return a;
}
TMCX mt& operator/=(mt& a, matScalarOf<mt> s) noexcept {
    for (std::size_t i = 0; i < matDimOf<mt>; ++i)
        for (std::size_t j = 0; j < matDimOf<mt>; ++j) a.m[i][j] /= s;
    return a;
}

/* binary */
TMNDCX mt operator+(mt a, const mt& b) noexcept { return a += b; }
TMNDCX mt operator-(mt a, const mt& b) noexcept { return a -= b; }
TMNDCX mt operator*(mt a, matScalarOf<mt> s) noexcept { return a *= s; }
TMNDCX mt operator*(matScalarOf<mt> s, mt a) noexcept { return a *= s; }
TMNDCX mt operator/(mt a, matScalarOf<mt> s) noexcept { return a /= s; }

TMNDCX bool operator==(const mt& a, const mt& b) noexcept {
    for (std::size_t i = 0; i < matDimOf<mt>; ++i)
        for (std::size_t j = 0; j < matDimOf<mt>; ++j)
            if (!(a.m[i][j] == b.m[i][j])) return false;
    return true;
}

TMFLT [[nodiscard]] constexpr bool approxEqual(const mt& a, const mt& b, matScalarOf<mt> eps = matScalarOf<mt>(1e-5)) noexcept {
    for (std::size_t i = 0; i < matDimOf<mt>; ++i)
        for (std::size_t j = 0; j < matDimOf<mt>; ++j)
            if (abs(a.m[i][j] - b.m[i][j]) > eps) return false;
    return true;
}
#pragma endregion
#pragma region matrix products
/* matrix * matrix: (a * b) applies b first, then a */
TMNDCX mt operator*(const mt& a, const mt& b) noexcept {
    mt r{};
    for (std::size_t i = 0; i < matDimOf<mt>; ++i)
        for (std::size_t j = 0; j < matDimOf<mt>; ++j) {
            matScalarOf<mt> sum{};
            for (std::size_t k = 0; k < matDimOf<mt>; ++k) sum += a.m[i][k] * b.m[k][j];
            r.m[i][j] = sum;
        }
    return r;
}
TMCX mt& operator*=(mt& a, const mt& b) noexcept { a = a * b; return a; }

/* matrix * column vector */
SCALTNDCX vec2<t> operator*(const matrix2x2<t>& a, const vec2<t>& v) noexcept {
    return { a.m[0][0] * v.x + a.m[0][1] * v.y,
             a.m[1][0] * v.x + a.m[1][1] * v.y };
}
SCALTNDCX vec3<t> operator*(const matrix3x3<t>& a, const vec3<t>& v) noexcept {
    return { a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
             a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
             a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z };
}
SCALTNDCX vec4<t> operator*(const matrix4x4<t>& a, const vec4<t>& v) noexcept {
    return { a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z + a.m[0][3] * v.w,
             a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z + a.m[1][3] * v.w,
             a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z + a.m[2][3] * v.w,
             a.m[3][0] * v.x + a.m[3][1] * v.y + a.m[3][2] * v.z + a.m[3][3] * v.w };
}

/* point: w = 1, bottom row ignored (affine matrices) */
SCALTNDCX vec3<t> transformPoint(const matrix4x4<t>& a, const vec3<t>& p) noexcept {
    return { a.m[0][0] * p.x + a.m[0][1] * p.y + a.m[0][2] * p.z + a.m[0][3],
             a.m[1][0] * p.x + a.m[1][1] * p.y + a.m[1][2] * p.z + a.m[1][3],
             a.m[2][0] * p.x + a.m[2][1] * p.y + a.m[2][2] * p.z + a.m[2][3] };
}
/* direction: w = 0, translation ignored */
SCALTNDCX vec3<t> transformVector(const matrix4x4<t>& a, const vec3<t>& v) noexcept {
    return { a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
             a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
             a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z };
}
/* full projective transform: w = 1, then divide by the resulting w (zero vector if w == 0) */
FLTTNDCX vec3<t> projectPoint(const matrix4x4<t>& a, const vec3<t>& p) noexcept {
    const vec4<t> r = a * vec4<t>{ p.x, p.y, p.z, t(1) };
    return r.w != t(0) ? vec3<t>{ r.x / r.w, r.y / r.w, r.z / r.w } : vec3<t>{};
}
#pragma endregion
#pragma region matrix transpose, determinant, trace
TMNDCX mt transpose(const mt& a) noexcept {
    mt r{};
    for (std::size_t i = 0; i < matDimOf<mt>; ++i)
        for (std::size_t j = 0; j < matDimOf<mt>; ++j) r.m[i][j] = a.m[j][i];
    return r;
}
TMNDCX matScalarOf<mt> trace(const mt& a) noexcept {
    matScalarOf<mt> s{};
    for (std::size_t i = 0; i < matDimOf<mt>; ++i) s += a.m[i][i];
    return s;
}

SCALTNDCX t determinant(const matrix2x2<t>& a) noexcept {
    return a.m[0][0] * a.m[1][1] - a.m[0][1] * a.m[1][0];
}
SCALTNDCX t determinant(const matrix3x3<t>& a) noexcept {
    return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1])
         - a.m[0][1] * (a.m[1][0] * a.m[2][2] - a.m[1][2] * a.m[2][0])
         + a.m[0][2] * (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]);
}
/* Laplace expansion along 2x2 sub-determinants of the top and bottom row pairs */
SCALTNDCX t determinant(const matrix4x4<t>& a) noexcept {
    const auto& m = a.m;
    const t s0 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    const t s1 = m[0][0] * m[1][2] - m[0][2] * m[1][0];
    const t s2 = m[0][0] * m[1][3] - m[0][3] * m[1][0];
    const t s3 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    const t s4 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
    const t s5 = m[0][2] * m[1][3] - m[0][3] * m[1][2];
    const t c5 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
    const t c4 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
    const t c3 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
    const t c2 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
    const t c1 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
    const t c0 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
    return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
}
#pragma endregion
#pragma region matrix conversion
/* upper-left 3x3 of a 4x4 (drops translation and the bottom row) */
SCALTNDCX matrix3x3<t> toMat3(const matrix4x4<t>& a) noexcept {
    return {{ { a.m[0][0], a.m[0][1], a.m[0][2] },
              { a.m[1][0], a.m[1][1], a.m[1][2] },
              { a.m[2][0], a.m[2][1], a.m[2][2] } }};
}
/* embeds a 3x3 in the upper-left of an identity 4x4 */
SCALTNDCX matrix4x4<t> toMat4(const matrix3x3<t>& a) noexcept {
    return {{ { a.m[0][0], a.m[0][1], a.m[0][2], t(0) },
              { a.m[1][0], a.m[1][1], a.m[1][2], t(0) },
              { a.m[2][0], a.m[2][1], a.m[2][2], t(0) },
              { t(0),      t(0),      t(0),      t(1) } }};
}
/* rotation matrix of a unit quaternion */
FLTTNDCX matrix3x3<t> toMat3(const quat<t>& q) noexcept {
    const t x = q.x, y = q.y, z = q.z, w = q.w;
    return {{ { t(1) - t(2) * (y * y + z * z), t(2) * (x * y - z * w),        t(2) * (x * z + y * w) },
              { t(2) * (x * y + z * w),        t(1) - t(2) * (x * x + z * z), t(2) * (y * z - x * w) },
              { t(2) * (x * z - y * w),        t(2) * (y * z + x * w),        t(1) - t(2) * (x * x + y * y) } }};
}
FLTTNDCX matrix4x4<t> toMat4(const quat<t>& q) noexcept { return toMat4(toMat3(q)); }

/* quaternion from a pure rotation matrix (Shepperd's method; stable for all rotations) */
FLTTND quat<t> toQuat(const matrix3x3<t>& a) noexcept {
    const auto& m = a.m;
    const t tr = m[0][0] + m[1][1] + m[2][2];
    quat<t> q{};
    if (tr > t(0)) {
        const t s = std::sqrt(tr + t(1)) * t(2);            // 4w
        q = { (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, s * t(0.25) };
    }
    else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const t s = std::sqrt(t(1) + m[0][0] - m[1][1] - m[2][2]) * t(2); // 4x
        q = { s * t(0.25), (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s };
    }
    else if (m[1][1] > m[2][2]) {
        const t s = std::sqrt(t(1) + m[1][1] - m[0][0] - m[2][2]) * t(2); // 4y
        q = { (m[0][1] + m[1][0]) / s, s * t(0.25), (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s };
    }
    else {
        const t s = std::sqrt(t(1) + m[2][2] - m[0][0] - m[1][1]) * t(2); // 4z
        q = { (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s * t(0.25), (m[1][0] - m[0][1]) / s };
    }
    return normalize(q);
}
/* uses the upper-left 3x3, which must be a pure rotation (no scale) */
FLTTND quat<t> toQuat(const matrix4x4<t>& a) noexcept { return toQuat(toMat3(a)); }
#pragma endregion
#pragma region matrix inverse
/* Each returns false (and leaves out untouched) if the matrix is singular. out may alias a. */
FLTTNDCX bool tryInverse(const matrix2x2<t>& a, matrix2x2<t>& out) noexcept {
    const t det = determinant(a);
    if (det == t(0)) return false;
    const t inv = t(1) / det;
    out = {{ {  a.m[1][1] * inv, -a.m[0][1] * inv },
             { -a.m[1][0] * inv,  a.m[0][0] * inv } }};
    return true;
}
/* inverse = cross products of the rows, divided by the determinant */
FLTTNDCX bool tryInverse(const matrix3x3<t>& a, matrix3x3<t>& out) noexcept {
    const vec3<t> r0{ a.m[0][0], a.m[0][1], a.m[0][2] };
    const vec3<t> r1{ a.m[1][0], a.m[1][1], a.m[1][2] };
    const vec3<t> r2{ a.m[2][0], a.m[2][1], a.m[2][2] };
    const vec3<t> c0 = cross(r1, r2), c1 = cross(r2, r0), c2 = cross(r0, r1);
    const t det = dot(r0, c0);
    if (det == t(0)) return false;
    const t inv = t(1) / det;
    out = {{ { c0.x * inv, c1.x * inv, c2.x * inv },
             { c0.y * inv, c1.y * inv, c2.y * inv },
             { c0.z * inv, c1.z * inv, c2.z * inv } }};
    return true;
}
FLTTNDCX bool tryInverse(const matrix4x4<t>& a, matrix4x4<t>& out) noexcept {
    const auto& m = a.m;
    const t s0 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    const t s1 = m[0][0] * m[1][2] - m[0][2] * m[1][0];
    const t s2 = m[0][0] * m[1][3] - m[0][3] * m[1][0];
    const t s3 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    const t s4 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
    const t s5 = m[0][2] * m[1][3] - m[0][3] * m[1][2];
    const t c5 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
    const t c4 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
    const t c3 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
    const t c2 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
    const t c1 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
    const t c0 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
    const t det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if (det == t(0)) return false;
    const t d = t(1) / det;
    out = {{
        { ( m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3) * d,
          (-m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3) * d,
          ( m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3) * d,
          (-m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3) * d },
        { (-m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1) * d,
          ( m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1) * d,
          (-m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1) * d,
          ( m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1) * d },
        { ( m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0) * d,
          (-m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0) * d,
          ( m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0) * d,
          (-m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0) * d },
        { (-m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0) * d,
          ( m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0) * d,
          (-m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0) * d,
          ( m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0) * d }
    }};
    return true;
}

/* General inverse. Returns identity if the matrix is singular (same fallback as inverse(quat));
   use tryInverse when you need to know. */
TMFLT [[nodiscard]] constexpr mt inverse(const mt& a) noexcept {
    mt r{};
    return tryInverse(a, r) ? r : matIdentity<mt>();
}

/* Faster inverse for affine matrices (bottom row 0 0 0 1), e.g. anything built from
   translation / rotation / scaling. Not valid for projection matrices. */
FLTTNDCX matrix4x4<t> inverseAffine(const matrix4x4<t>& a) noexcept {
    const matrix3x3<t> r = inverse(toMat3(a));
    const vec3<t> p{ a.m[0][3], a.m[1][3], a.m[2][3] };
    const vec3<t> np = r * p;
    matrix4x4<t> out = toMat4(r);
    out.m[0][3] = -np.x;
    out.m[1][3] = -np.y;
    out.m[2][3] = -np.z;
    return out;
}

/* matrix for transforming normals: transpose(inverse(upper 3x3)). Stays correct under non-uniform scale. */
FLTTNDCX matrix3x3<t> normalMatrix(const matrix4x4<t>& a) noexcept {
    return transpose(inverse(toMat3(a)));
}
#pragma endregion
#pragma region matrix factories
FLTTNDCX matrix4x4<t> translation(const vec3<t>& p) noexcept {
    matrix4x4<t> r = matIdentity<matrix4x4<t>>();
    r.m[0][3] = p.x; r.m[1][3] = p.y; r.m[2][3] = p.z;
    return r;
}
FLTTNDCX matrix4x4<t> scaling(const vec3<t>& s) noexcept {
    matrix4x4<t> r{};
    r.m[0][0] = s.x; r.m[1][1] = s.y; r.m[2][2] = s.z; r.m[3][3] = t(1);
    return r;
}
FLTTNDCX matrix4x4<t> scaling(t s) noexcept { return scaling(vec3<t>{ s, s, s }); }

/* counter-clockwise rotation about an axis (radians); right-handed */
FLTTND matrix4x4<t> rotationX(t angle) noexcept {
    const t c = std::cos(angle), s = std::sin(angle);
    return {{ { t(1), t(0), t(0), t(0) },
              { t(0), c,    -s,   t(0) },
              { t(0), s,    c,    t(0) },
              { t(0), t(0), t(0), t(1) } }};
}
FLTTND matrix4x4<t> rotationY(t angle) noexcept {
    const t c = std::cos(angle), s = std::sin(angle);
    return {{ { c,    t(0), s,    t(0) },
              { t(0), t(1), t(0), t(0) },
              { -s,   t(0), c,    t(0) },
              { t(0), t(0), t(0), t(1) } }};
}
FLTTND matrix4x4<t> rotationZ(t angle) noexcept {
    const t c = std::cos(angle), s = std::sin(angle);
    return {{ { c,    -s,   t(0), t(0) },
              { s,    c,    t(0), t(0) },
              { t(0), t(0), t(1), t(0) },
              { t(0), t(0), t(0), t(1) } }};
}
/* axis need not be normalized */
FLTTND matrix4x4<t> rotation(const vec3<t>& axis, t angle) noexcept {
    return toMat4(quat<t>::fromAxisAngle(axis, angle));
}
/* 2D counter-clockwise rotation (radians) */
FLTTND matrix2x2<t> rotation2D(t angle) noexcept {
    const t c = std::cos(angle), s = std::sin(angle);
    return {{ { c, -s },
              { s,  c } }};
}

/* Splits an affine T*R*S matrix back into position / rotation / scale. Exact when there is no shear.
   A mirrored matrix (negative determinant) is folded into scale.x. If any axis has zero scale the
   rotation can't be recovered and is left as identity. */
FLTTND transform<t> decompose(const matrix4x4<t>& a) noexcept {
    transform<t> out{};
    out.position = { a.m[0][3], a.m[1][3], a.m[2][3] };
    const vec3<t> c0{ a.m[0][0], a.m[1][0], a.m[2][0] };
    const vec3<t> c1{ a.m[0][1], a.m[1][1], a.m[2][1] };
    const vec3<t> c2{ a.m[0][2], a.m[1][2], a.m[2][2] };
    vec3<t> s{ length(c0), length(c1), length(c2) };
    if (determinant(toMat3(a)) < t(0)) s.x = -s.x;
    out.scale = s;
    if (s.x != t(0) && s.y != t(0) && s.z != t(0)) {
        const vec3<t> r0 = c0 / s.x, r1 = c1 / s.y, r2 = c2 / s.z;
        out.rotation = toQuat(fromColumns(r0, r1, r2));
    }
    return out;
}
#pragma endregion
#pragma region camera and projection
/* view matrix: eye looks at target, up is the approximate up direction.
   Degenerate (singular) if target == eye or the view direction is parallel to up. */
FLTTND matrix4x4<t> lookAt(const vec3<t>& eye, const vec3<t>& target, const vec3<t>& up) noexcept {
    const vec3<t> f = normalize(target - eye);
    const vec3<t> s = normalize(cross(f, up));
    const vec3<t> u = cross(s, f);
    return {{ {  s.x,  s.y,  s.z, -dot(s, eye) },
              {  u.x,  u.y,  u.z, -dot(u, eye) },
              { -f.x, -f.y, -f.z,  dot(f, eye) },
              { t(0), t(0), t(0),  t(1) } }};
}

/* fovY in radians. OpenGL clip space: z in [-1,1] */
FLTTND matrix4x4<t> perspective(t fovY, t aspect, t zNear, t zFar) noexcept {
    const t f = t(1) / std::tan(fovY * t(0.5));
    matrix4x4<t> r{};
    r.m[0][0] = f / aspect;
    r.m[1][1] = f;
    r.m[2][2] = -(zFar + zNear) / (zFar - zNear);
    r.m[2][3] = -(t(2) * zFar * zNear) / (zFar - zNear);
    r.m[3][2] = t(-1);
    return r;
}
/* zero-to-one depth (Vulkan / D3D / Metal) */
FLTTND matrix4x4<t> perspectiveZO(t fovY, t aspect, t zNear, t zFar) noexcept {
    const t f = t(1) / std::tan(fovY * t(0.5));
    matrix4x4<t> r{};
    r.m[0][0] = f / aspect;
    r.m[1][1] = f;
    r.m[2][2] = zFar / (zNear - zFar);
    r.m[2][3] = (zFar * zNear) / (zNear - zFar);
    r.m[3][2] = t(-1);
    return r;
}

/* OpenGL clip space: z in [-1,1] */
FLTTNDCX matrix4x4<t> orthographic(t left, t right, t bottom, t top, t zNear, t zFar) noexcept {
    matrix4x4<t> r{};
    r.m[0][0] = t(2) / (right - left);
    r.m[1][1] = t(2) / (top - bottom);
    r.m[2][2] = t(-2) / (zFar - zNear);
    r.m[0][3] = -(right + left) / (right - left);
    r.m[1][3] = -(top + bottom) / (top - bottom);
    r.m[2][3] = -(zFar + zNear) / (zFar - zNear);
    r.m[3][3] = t(1);
    return r;
}
/* zero-to-one depth */
FLTTNDCX matrix4x4<t> orthographicZO(t left, t right, t bottom, t top, t zNear, t zFar) noexcept {
    matrix4x4<t> r = orthographic(left, right, bottom, top, zNear, zFar);
    r.m[2][2] = t(-1) / (zFar - zNear);
    r.m[2][3] = -zNear / (zFar - zNear);
    return r;
}
#pragma endregion
#pragma endregion
#pragma region vectorMathExtension
#pragma region traits
template<typename v> struct vecTraits {};

template<scalar t> struct vecTraits<vec2<t>> {
    using scalar_type = t;
    static constexpr std::size_t dim = 2;
    template<scalar u> using rebind = vec2<u>;
};
template<scalar t> struct vecTraits<vec3<t>> {
    using scalar_type = t;
    static constexpr std::size_t dim = 3;
    template<scalar u> using rebind = vec3<u>;
};
template<scalar t> struct vecTraits<vec4<t>> {
    using scalar_type = t;
    static constexpr std::size_t dim = 4;
    template<scalar u> using rebind = vec4<u>;
};


TV using scalarOf = typename vecTraits<v>::scalar_type;
TV inline constexpr std::size_t dimOf = vecTraits<v>::dim;
#pragma endregion
#pragma region component-wise arithmetic
/* vector * vector and vector / vector multiply or divide per component (like GLSL) */
TVNDCX v operator*(const v& a, const v& b) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = a[i] * b[i]; return r;
}
TVNDCX v operator/(const v& a, const v& b) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = a[i] / b[i]; return r;
}
TVCX v& operator*=(v& a, const v& b) noexcept { a = a * b; return a; }
TVCX v& operator/=(v& a, const v& b) noexcept { a = a / b; return a; }
#pragma endregion
#pragma region component-wise general
TVNDCX v min(const v& a, const v& b) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = min(a[i], b[i]); return r;
}
TVNDCX v max(const v& a, const v& b) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = max(a[i], b[i]); return r;
}
/* per-component bounds */
TVNDCX v clamp(const v& x, const v& lo, const v& hi) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = clamp(x[i], lo[i], hi[i]); return r;
}
/* same bounds for every component */
TVNDCX v clamp(const v& x, scalarOf<v> lo, scalarOf<v> hi) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = clamp(x[i], lo, hi); return r;
}
TVNDCX v abs(const v& a) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = abs(a[i]); return r;
}
TVNDCX v sign(const v& a) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = sign(a[i]); return r;
}

/* floating-point only */
template<vector_type v> requires std::floating_point<scalarOf<v>>
NDCX v saturate(const v& a) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = saturate(a[i]); return r;
}
template<vector_type v> requires std::floating_point<scalarOf<v>>
NDCX v ceil(const v& a) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = ceil(a[i]); return r;
}
template<vector_type v> requires std::floating_point<scalarOf<v>>
NDCX v trunc(const v& a) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = trunc(a[i]); return r;
}
template<vector_type v> requires std::floating_point<scalarOf<v>>
NDCX v round(const v& a) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = round(a[i]); return r;
}
template<vector_type v> requires std::floating_point<scalarOf<v>>
NDCX v fract(const v& a) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = fract(a[i]); return r;
}

/* same blend factor for every component; not clamped */
template<vector_type v> requires std::floating_point<scalarOf<v>>
NDCX v lerp(const v& a, const v& b, scalarOf<v> f) noexcept {
    v r{}; for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = lerp(a[i], b[i], f); return r;
}
/* smallest / largest single component */
TVNDCX scalarOf<v> minComponent(const v& a) noexcept {
    scalarOf<v> m = a[0]; for (std::size_t i = 1; i < dimOf<v>; ++i) m = min(m, a[i]); return m;
}
TVNDCX scalarOf<v> maxComponent(const v& a) noexcept {
    scalarOf<v> m = a[0]; for (std::size_t i = 1; i < dimOf<v>; ++i) m = max(m, a[i]); return m;
}
#pragma endregion
#pragma region geometric
TVNDCX scalarOf<v> distanceSquared(const v& a, const v& b) noexcept { return lengthSquared(b - a); }

template<vector_type v> requires std::floating_point<scalarOf<v>>
[[nodiscard]] scalarOf<v> distance(const v& a, const v& b) noexcept { return length(b - a); }

/* reflects incident vector i about the surface normal n (n must be normalized) */
template<vector_type v> requires std::floating_point<scalarOf<v>>
NDCX v reflect(const v& i, const v& n) noexcept {
    using s = scalarOf<v>;
    return i - n * (s(2) * dot(n, i));
}

/* refracts i through a surface with normal n and index ratio eta (both i and n normalized).
   Returns the zero vector on total internal reflection. */
template<vector_type v> requires std::floating_point<scalarOf<v>>
[[nodiscard]] v refract(const v& i, const v& n, scalarOf<v> eta) noexcept {
    using s = scalarOf<v>;
    const s d = dot(n, i);
    const s k = s(1) - eta * eta * (s(1) - d * d);
    if (k < s(0)) return v{};
    return i * eta - n * (eta * d + std::sqrt(k));
}

/* component of a that lies along onto (zero vector if onto is zero) */
template<vector_type v> requires std::floating_point<scalarOf<v>>
NDCX v project(const v& a, const v& onto) noexcept {
    using s = scalarOf<v>;
    const s d = dot(onto, onto);
    return d > s(0) ? onto * (dot(a, onto) / d) : v{};
}

/* component of a perpendicular to onto */
template<vector_type v> requires std::floating_point<scalarOf<v>>
NDCX v reject(const v& a, const v& onto) noexcept { return a - project(a, onto); }

/* unsigned angle in radians between a and b, in [0, pi]; 0 if either is the zero vector */
template<vector_type v> requires std::floating_point<scalarOf<v>>
[[nodiscard]] scalarOf<v> angleBetween(const v& a, const v& b) noexcept {
    using s = scalarOf<v>;
    const s denom = length(a) * length(b);
    return denom > s(0) ? std::acos(clamp(dot(a, b) / denom, s(-1), s(1))) : s(0);
}
#pragma endregion
#pragma region vec2 / vec3 specific
/* 90 degrees counter-clockwise */
SCALTNDCX vec2<t> perpendicular(const vec2<t>& v) noexcept { return { -v.y, v.x }; }

/* counter-clockwise rotation by angle (radians) */
FLTTND vec2<t> rotate(const vec2<t>& v, t angle) noexcept {
    const t c = std::cos(angle), s = std::sin(angle);
    return { v.x * c - v.y * s, v.x * s + v.y * c };
}

/* unit vector at angle (radians) from the +x axis */
FLTTND vec2<t> fromAngle(t angle) noexcept { return { std::cos(angle), std::sin(angle) }; }

/* some vector perpendicular to v (not normalized; zero only if v is zero) */
FLTTNDCX vec3<t> orthogonal(const vec3<t>& v) noexcept {
    return abs(v.x) > abs(v.z) ? vec3<t>{ -v.y, v.x, t(0) } : vec3<t>{ t(0), -v.z, v.y };
}
#pragma endregion
#pragma region conversion
/* converts components to another scalar type: vecCast<int>(vec3<float>{...}) -> vec3<int> */
template<scalar u, vector_type v>
NDCX typename vecTraits<v>::template rebind<u> vecCast(const v& a) noexcept {
    typename vecTraits<v>::template rebind<u> r{};
    for (std::size_t i = 0; i < dimOf<v>; ++i) r[i] = static_cast<u>(a[i]);
    return r;
}
#pragma endregion
#pragma region vectorTypedefs
typedef vec2<float> vec2f; typedef vec2<double> vec2d; typedef vec2<int> vec2i; typedef vec2<unsigned> vec2u;
typedef vec3<float> vec3f; typedef vec3<double> vec3d; typedef vec3<int> vec3i; typedef vec3<unsigned> vec3u;
typedef vec4<float> vec4f; typedef vec4<double> vec4d; typedef vec4<int> vec4i; typedef vec4<unsigned> vec4u;
#pragma endregion
#pragma endregion
#pragma endregion