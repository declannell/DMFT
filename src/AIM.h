#pragma once
#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
#include <eigen3/Eigen/Dense>

typedef std::complex<double> dcomp;

class AIM
{
public:
    std::vector<dcomp> impurity_gf_mb_retarded;
    std::vector<dcomp> dynamical_field_retarded;
    std::vector<dcomp> self_energy_mb_retarded;
    
    std::vector<double> self_energy_mb_lesser;
    std::vector<double> impurity_gf_mb_lesser;
    std::vector<double> hybridisation_lesser;
    std::vector<double> dynamical_field_lesser;
    
    std::vector<double> self_energy_mb_greater;
    std::vector<double> impurity_gf_mb_greater;
    std::vector<double> hybridisation_greater;
    std::vector<double> dynamical_field_greater;

    std::vector<double> fermi_function_eff;

    AIM(const Parameters &parameters, const std::vector<dcomp> &local_gf_retarded, const std::vector<dcomp> &local_gf_lesser, 
    const std::vector<dcomp> &local_gf_greater, const std::vector<dcomp> &self_energy_retarded, const std::vector<dcomp> &self_energy_lesser,
    const std::vector<dcomp> &self_energy_greater, const int voltage_step);
    

    void get_impurity_gf_mb(const Parameters &parameters, const std::vector<dcomp> &local_gf_retarded, const std::vector<dcomp> &local_gf_lesser);

    void get_impurity_gf_mb(const Parameters &parameters, const std::vector<dcomp> &local_gf_retarded,
         const std::vector<dcomp> &local_gf_lesser, const std::vector<dcomp> &local_gf_greater);

    void get_retarded_dynamical_field(const Parameters &parameters, const std::vector<dcomp> &local_gf_retarded, const std::vector<dcomp> &self_energy_retarded);

    void get_effective_fermi_function(const Parameters &parameters, const int voltage_step);

    void get_lesser_hybridisation(const Parameters &parameters, const std::vector<dcomp> &self_energy_lesser);

    void get_lesser_greater_hybridisation (const Parameters &parameters, const std::vector<dcomp> &self_energy_lesser, const std::vector<dcomp> &self_energy_greater);

    void get_dynamical_field_lesser_greater(const Parameters &parameters, const int voltage_step);

    void get_dynamical_field_lesser(const Parameters &parameters, const int voltage_step);
};