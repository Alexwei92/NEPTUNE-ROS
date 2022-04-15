#include <math.h>
#include "math_utils.h"
#include "rulebased_ctrl.h"

float rulebased_ctrl(myAffordance affordance) {
    float command = 0.0;

    float dist_center_width = affordance.dist_center_width;
    float dist_left_width = affordance.dist_center_width - 0.5;
    float rel_angle = affordance.rel_angle / (M_PI_2);

    // tunable parameters
    float C_1 = 20.0; 
    float C_2 = 1.5; 

    command = 1.0 / (1.0 + exp(C_1 * (C_2 * rel_angle / M_PI + dist_center_width)));
    command = 1.0 * (2.0 * command - 1.0); // -> [-1,1]
    
    return constrain_float(command, -1.0f, 1.0f);
}