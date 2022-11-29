#include "pid.h"

using namespace std;

// constructor
PID::PID(float kp=1.0f, float ki=0.0f, float kd=0.0f, float dt=0.05f,
		 float integral_limit=INF, float output_limit=INF) 
{
	// set parameters
	_kp = kp;
	_ki = ki;
	_kd = kd;
	_dt = fabsf(dt);
	_integral_limit = fabsf(integral_limit);
	_output_limit = fabsf(output_limit);

	// set LPF
	_alpha_derivative = calc_lowpass_alpha_dt(_dt, 10.0f); // cutoff_freq = 10 Hz

	// init variables
	PID::reset();
};

// calculate PID output
float PID::calculate(float target, float current)
{
	if (!isfinite(target) || !isfinite(current)) {
		return 0.0f;
	}
	
	// error
	_error = target - current;

	// calculate derivative term
	float derivative = (_error - _error_previous) / _dt;
	if (!isfinite(derivative)) {
		derivative = 0.0f;
	}
	_derivative += _alpha_derivative * (derivative - _derivative);

	// calculate integral term
	_integral += (_error * _dt);
	if (isfinite(_integral_limit) && _integral_limit != 0.0f) {
		_integral = constrain_float(_integral, -_integral_limit, _integral_limit);
	}
	
	// total output
	float output = (_error * _kp) + (_integral * _ki) + (_derivative * _kd);

	// limit output
	if (isfinite(_output_limit) && _output_limit != 0.0f) {
		output = constrain_float(output, -_output_limit, _output_limit);
	}

	return output;
}

// reset integral
void PID::reset_integral()
{
	_integral = 0.0f;
}

// reset all
void PID::reset()
{
	_error = 0.0f;
	_error_previous = 0.0f;
	_derivative = 0.0f;
	_integral = 0.0f;
}