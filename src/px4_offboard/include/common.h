#pragma once

#define CH_MIN  982 // minimum RC channel value
#define CH_MAX  2006 // maximum RC channel value
#define YAW_CHANNEL 3 // yaw channel index

#include "user_config.h"
 
/*
Normalize RCIn value to [-1, 1]
*/
float rc_mapping(uint16_t value) {
    float result = ((float)(value) - CH_MIN) / (CH_MAX - CH_MIN); 
    result = result * 2.0 - 1.0;
    return result;
}