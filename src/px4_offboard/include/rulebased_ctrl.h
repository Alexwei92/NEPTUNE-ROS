#pragma once

/** Affordance structure **/
struct myAffordance {
    float dist_center_width;
    float dist_left_width;
    float rel_angle;
};

/** Rule-based controller **/
float rulebased_ctrl(myAffordance affordance);