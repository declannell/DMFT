#pragma once
#include "parameters.h"
#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
//#include <eigen3/Eigen/Dense>

typedef std::complex<double> dcomp;

class Pseudo_GF
{
private:
    double kx_value, ky_value;

public:
    std::vector<dcomp> retarded_gf, retarded_se;
    std::vector<double> greater_gf, lesser_gf, greater_se, lesser_se;
    
    
    Pseudo_GF(const Parameters &parameters);

    //void get_interacting_gf(const Parameters &parameters);

    void print_green_function(const Parameters &parameters, int voltage_step, const std::string& name);

};


