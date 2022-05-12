#pragma once
#include "parameters.h"
#include "leads_self_energy.h"
#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
#include <eigen3/Eigen/Dense>

typedef std::complex<double> dcomp;

class Interacting_GF
{
private:
    double kx_value, ky_value;

public:
    std::vector<Eigen::MatrixXcd> interacting_gf;
    Interacting_GF(Parameters &parameters, double _kx, double _ky, std::vector<std::vector<dcomp>> &self_energy_mb, std::vector<dcomp> &self_energy_left,
                   std::vector<dcomp> &self_energy_right);

    Eigen::MatrixXcd get_hamiltonian(Parameters const &parameters);
    void get_interacting_gf(Parameters &parameters, const Eigen::MatrixXcd& hamiltonian, std::vector<std::vector<dcomp>> const &self_energy_mb, 
                            std::vector<dcomp> const &self_energy_left, std::vector<dcomp> const &self_energy_right);
    double kx() const;
    double ky() const;

};

void get_analytic_gf_1_site(Parameters &parameters, std::vector<Eigen::MatrixXcd> green_function);
void run(Parameters &parameters);

void get_local_gf(Parameters &parameters, std::vector<double> const &kx, std::vector<double> const &ky, std::vector<std::vector<dcomp>> &self_energy_mb, 
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &gf_local_down);



