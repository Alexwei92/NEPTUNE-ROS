#pragma once

#include <limits>
#include <cmath>
#include "math_utils.h"

#define INF std::numeric_limits<float>::infinity()

class PID
{
public:
	/* Constructor for PID */
	PID(float kp, float ki, float kd, float dt,
		float integral_limit, float output_limit);

	~PID() {};

	/* Calculate PID output */
	float calculate(float target, float current);

	/* Get PID gains */
	float get_kp() const { return _kp; }
	float get_ki() const { return _ki; }
	float get_kd() const { return _kd; }
	float get_error() const { return _error; }
	float get_derivative() const { return _derivative; }
	float get_integral() const { return _integral; }

	/* Reset integral */
	void reset_integral();
	
	/* Reset all */
	void reset();

protected:
	float _kp; 				 // proportional gain
	float _ki; 				 // integral gain
	float _kd; 				 // derivative gain
	float _dt; 				 // sample time
	float _integral_limit;   // integral limit
	float _output_limit; 	 // output limit
	float _alpha_derivative; // low pass filter coefficient for derivative term

private:
	float _error;            // error
	float _error_previous; 	 // previous error
	float _derivative;       // derivative value
	float _integral; 		 // integral value
};


inline float calc_lowpass_alpha_dt(float dt, float cutoff_freq)
{
	if (dt <= 0.0f || cutoff_freq <= 0.0f) {
		return 1.0f;
	}
	float rc = 1.0f / (M_2PI * cutoff_freq);
	return dt / (dt + rc);
}