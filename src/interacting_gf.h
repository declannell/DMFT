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
    Interacting_GF(const Parameters &parameters, const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<Eigen::MatrixXcd> 
    &self_energy_left, const std::vector<Eigen::MatrixXcd> &self_energy_right, const int voltage_step, const  Eigen::MatrixXcd &hamiltonian);

    void get_interacting_gf(const Parameters &parameters, const Eigen::MatrixXcd& hamiltonian, const std::vector<std::vector<dcomp>> &self_energy_mb, 
        const std::vector<Eigen::MatrixXcd> &self_energy_left, const std::vector<Eigen::MatrixXcd> &self_energy_right, const int voltage_step);

    double kx() const;
    double ky() const;

};

void get_hamiltonian(Parameters const &parameters, const int voltage_step, const double kx, const double ky, Eigen::MatrixXcd &hamiltonian);

void get_analytic_gf_1_site(Parameters &parameters, std::vector<Eigen::MatrixXcd> &green_function, int voltage_step);

void run(Parameters &parameters);

void get_local_gf(Parameters &parameters, std::vector<std::vector<dcomp>> &self_energy_mb, 
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local, int voltage_step, const  std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian);

void get_advance_gf(const Parameters &parameters, const Eigen::MatrixXcd &gf_retarded, Eigen::MatrixXcd &gf_advanced);

void get_embedding_lesser(const Parameters &parameters, const Eigen::MatrixXcd &self_energy_left, 
    const Eigen::MatrixXcd &self_energy_right, Eigen::MatrixXcd &embedding_self_energy_lesser, int r, int voltage_step);

void get_embedding_greater(const Parameters &parameters, const Eigen::MatrixXcd &self_energy_left, 
    const Eigen::MatrixXcd &self_energy_right, Eigen::MatrixXcd &embedding_self_energy_greater, int r, int voltage_step);

void get_gf_lesser_non_eq(const Parameters &parameters, const std::vector<Eigen::MatrixXcd> &gf_retarded, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const std::vector<Eigen::MatrixXcd> &self_energy_left, 
    const std::vector<Eigen::MatrixXcd> &self_energy_right, std::vector<Eigen::MatrixXcd> &gf_lesser, int voltage_step);

void get_gf_lesser_greater_non_eq(const Parameters &parameters, const std::vector<Eigen::MatrixXcd> &gf_retarded, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const std::vector<std::vector<dcomp>> &self_energy_mb_greater,
    const std::vector<Eigen::MatrixXcd> &self_energy_left, const std::vector<Eigen::MatrixXcd> &self_energy_right, std::vector<Eigen::MatrixXcd> &gf_lesser,
    std::vector<Eigen::MatrixXcd> &gf_greater, int voltage_step);

void get_gf_lesser_fd(const Parameters &parameters, const std::vector<Eigen::MatrixXcd> &gf_retarded, std::vector<Eigen::MatrixXcd> &gf_lesser);

void get_local_gf_r_and_lesser(const Parameters &parameters,  
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local, 
    std::vector<Eigen::MatrixXcd> &gf_local_lesser, const int voltage_step, const std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian);

void get_local_gf_r_greater_lesser(const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<dcomp>> &self_energy_mb_greater,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local, 
    std::vector<Eigen::MatrixXcd> &gf_local_lesser, std::vector<Eigen::MatrixXcd> &gf_local_greater, 
    const int voltage_step, const std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian);

void get_noneq_test(const Parameters &parameters, const std::vector<std::vector<dcomp>> &self_energy_mb, 
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local, 
    std::vector<Eigen::MatrixXcd> &gf_local_lesser, const int voltage_step, const std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian);



