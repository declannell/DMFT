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
    Interacting_GF(const Parameters &parameters, const double _kx, const double _ky, const std::vector<std::vector<dcomp>> &self_energy_mb, const  std::vector<dcomp> &self_energy_left,
                const std::vector<dcomp> &self_energy_right, const int voltage_step);

    Eigen::MatrixXcd get_hamiltonian(const Parameters &parameters, const int voltage_step);
    void get_interacting_gf(const Parameters &parameters, const Eigen::MatrixXcd& hamiltonian, const std::vector<std::vector<dcomp>> &self_energy_mb, 
                            std::vector<dcomp> const &self_energy_left, std::vector<dcomp> const &self_energy_right, const int voltage_step);
    double kx() const;
    double ky() const;

};

void get_analytic_gf_1_site(Parameters &parameters, std::vector<Eigen::MatrixXcd> &green_function, int voltage_step);
void run(Parameters &parameters);

void get_local_gf(Parameters &parameters, std::vector<double> const &kx, std::vector<double> const &ky, std::vector<std::vector<dcomp>> &self_energy_mb, 
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local, int voltage_step);

void get_advance_gf(const Parameters &parameters, const Eigen::MatrixXcd &gf_retarded, Eigen::MatrixXcd &gf_advanced);

void get_gf_lesser_non_eq(const Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_retarded, 
    std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const std::vector<dcomp> &self_energy_left,
    const std::vector<dcomp> &self_energy_right, std::vector<Eigen::MatrixXcd> &gf_lesser_local, int voltage_step);

void get_local_gf_r_and_lesser(Parameters &parameters, std::vector<double> const &kx, std::vector<double> const &ky, 
    std::vector<std::vector<dcomp>> &self_energy_mb, std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local, 
    std::vector<Eigen::MatrixXcd> &gf_local_lesser, int voltage_step);



