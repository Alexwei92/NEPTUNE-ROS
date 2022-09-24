#pragma once

/*
Normalize RCIn value to [-1, 1]
*/
float rc_mapping(uint16_t value, uint16_t pwm_min, uint16_t pwm_max) {
    float result = ((float)(value) - pwm_min) / (pwm_max - pwm_min); 
    result = result * 2.0 - 1.0;
    return result;
}