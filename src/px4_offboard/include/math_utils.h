#pragma once

#ifdef M_PI
# undef M_PI
#endif
#define M_PI      (3.141592653589793)

#ifdef M_PI_2
# undef M_PI_2
#endif
#define M_PI_2    (M_PI / 2)

#ifdef M_2PI
# undef M_2PI
#endif
#define M_2PI     (M_PI * 2)

#define RAD2DEG     (180.0 / M_PI)
#define DEG2RAD     (M_PI / 180.0)

template <typename T>
T constrain_value(const T value, const T low, const T high)
{
    if (value < low) {
        return low;
    }

    if (value > high) {
        return high;
    }

    return value;
}

inline float constrain_float(const float value, const float low, const float high)
{
    return constrain_value(value, low, high);
}

float wrap_360(const float angle)
{
    float res = fmodf(angle, 360.0f);
    if (res < 0) {
        res += 360.0f;
    }
    return res;
}

template <typename T>
T wrap_180(const T angle)
{
    auto res = wrap_360(angle);
    if (res > T(180)) {
        res -= T(360);
    }
    return res;
}

float wrap_2PI(const float radian)
{
    float res = fmodf(radian, M_2PI);
    if (res < 0) {
        res += M_2PI;
    }
    return res;
}

float wrap_PI(const float radian)
{
    float res = wrap_2PI(radian);
    if (res > M_PI) {
        res -= M_2PI;
    }
    return res;
}