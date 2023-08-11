#pragma once
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include "dmft.h"
#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"
#include "transport.h"


void integrate_spectral(Parameters &parameters, MatrixVectorType const &self_energy_left, MatrixVectorType const &self_energy_right);
void analytic_gf(Parameters &parameters, MatrixVectorType &gf_local);
