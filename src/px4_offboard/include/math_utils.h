#pragma once

#include <cmath>

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

inline int16_t constrain_int16(const int16_t value, const int16_t low, const int16_t high)
{
    return constrain_value(value, low, high);
}

inline int32_t constrain_int32(const int32_t value, const int32_t low, const int32_t high)
{
    return constrain_value(value, low, high);
}

inline int64_t constrain_int64(const int64_t value, const int64_t low, const int64_t high)
{
    return constrain_value(value, low, high);
}