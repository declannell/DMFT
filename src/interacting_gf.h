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
    MatrixVectorType interacting_gf;
    Interacting_GF(const Parameters &parameters, const std::vector<std::vector<dcomp>> &self_energy_mb, const MatrixVectorType 
    &self_energy_left, const MatrixVectorType &self_energy_right, const int voltage_step, const  MatrixType &hamiltonian);

    void get_interacting_gf(const Parameters &parameters, const MatrixType& hamiltonian, const std::vector<std::vector<dcomp>> &self_energy_mb, 
        const MatrixVectorType &self_energy_left, const MatrixVectorType &self_energy_right, const int voltage_step);

    double kx() const;
    double ky() const;

};

void get_hamiltonian(Parameters const &parameters, const int voltage_step, const double kx, const double ky, MatrixType &hamiltonian, int spin);

void get_analytic_gf_1_site(Parameters &parameters, MatrixVectorType &green_function, int voltage_step);

void run(Parameters &parameters);

void get_local_gf(Parameters &parameters, std::vector<std::vector<dcomp>> &self_energy_mb, 
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, MatrixVectorType &gf_local, int voltage_step, const  std::vector<MatrixVectorType> &hamiltonian);

void get_advance_gf(const Parameters &parameters, const MatrixType &gf_retarded, MatrixType &gf_advanced);

void get_embedding_lesser(const Parameters &parameters, const MatrixType &self_energy_left, 
    const MatrixType &self_energy_right, MatrixType &embedding_self_energy_lesser, int r, int voltage_step);

void get_embedding_greater(const Parameters &parameters, const MatrixType &self_energy_left, 
    const MatrixType &self_energy_right, MatrixType &embedding_self_energy_greater, int r, int voltage_step);

void get_gf_lesser_non_eq(const Parameters &parameters, const MatrixVectorType &gf_retarded, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const MatrixVectorType &self_energy_left, 
    const MatrixVectorType &self_energy_right, MatrixVectorType &gf_lesser, int voltage_step);

void get_gf_lesser_greater_non_eq(const Parameters &parameters, const MatrixVectorType &gf_retarded, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const std::vector<std::vector<dcomp>> &self_energy_mb_greater,
    const MatrixVectorType &self_energy_left, const MatrixVectorType &self_energy_right, MatrixVectorType &gf_lesser,
    MatrixVectorType &gf_greater, int voltage_step);

void get_gf_lesser_fd(const Parameters &parameters, const MatrixVectorType &gf_retarded, MatrixVectorType &gf_lesser);

void get_local_gf_r_and_lesser(const Parameters &parameters,  
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, MatrixVectorType &gf_local, 
    MatrixVectorType &gf_local_lesser, const int voltage_step, const std::vector<MatrixVectorType> &hamiltonian);

void get_local_gf_r_greater_lesser(const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<dcomp>> &self_energy_mb_greater,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, MatrixVectorType &gf_local, 
    MatrixVectorType &gf_local_lesser, MatrixVectorType &gf_local_greater, 
    const int voltage_step, const std::vector<MatrixVectorType> &hamiltonian);

void get_noneq_test(const Parameters &parameters, const std::vector<std::vector<dcomp>> &self_energy_mb, 
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, MatrixVectorType &gf_local, 
    MatrixVectorType &gf_local_lesser, const int voltage_step, const std::vector<MatrixVectorType> &hamiltonian);



