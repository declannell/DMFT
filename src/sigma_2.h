#pragma once
#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include "AIM.h"
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>



double get_prefactor(const int i, const int j, const int r, const int voltage_step, const Parameters &parameters,
	 std::vector<double> &fermi_up, std::vector<double> &fermi_down);

double integrate_equilibrium(const Parameters& parameters, const std::vector<double>& gf_1, const std::vector<double>& gf_2, const std::vector<double>& gf_3, const int r,
    std::vector<double> &fermi_up, std::vector<double> &fermi_down, int voltage_step);

dcomp integrate(const Parameters& parameters, const std::vector<dcomp>& gf_1, const std::vector<dcomp>& gf_2, const std::vector<dcomp>& gf_3, const int r);

void self_energy_2nd_order(const Parameters& parameters, AIM &aim_up, AIM &aim_down);

void self_energy_2nd_order_kramers_kronig(const Parameters& parameters, AIM &aim_up, AIM &aim_down, const int voltage_step);
