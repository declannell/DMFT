#pragma once
#include <mpi.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include "utilis.h"
#include "dmft.h"
#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"

void get_k_dependent_transmission(const Parameters& parameters, MatrixVectorType& green_function,
    const MatrixVectorType &coupling_left, const MatrixVectorType &coupling_right, std::vector<dcomp> &transmission);

void get_transmission_gf_local(
    const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_up,
	const std::vector<std::vector<dcomp>> &self_energy_mb_down,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads,
    std::vector<dcomp> &transmission_up, std::vector<dcomp> &transmission_down,
    const int voltage_step, std::vector<MatrixVectorType> &hamiltonian_up,
    std::vector<MatrixVectorType> &hamiltonian_down, MatrixVectorType &gf_local_up, 
    MatrixVectorType &gf_local_lesser_up, MatrixVectorType &gf_local_down, 
    MatrixVectorType &gf_local_lesser_down);

void get_coupling(const Parameters &parameters, const MatrixVectorType &self_energy_left, const MatrixVectorType &self_energy_right, 
	MatrixVectorType &coupling_left, MatrixVectorType &coupling_right);

void get_landauer_buttiker_current(const Parameters& parameters,
    const std::vector<dcomp>& transmission_up, const std::vector<dcomp>& transmission_down,
    double* current_up, double* current_down, const int votlage_step);

void get_meir_wingreen_current(
    const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, const int voltage_step, double *current_left,
	double *current_right, std::vector<dcomp>& transmission_up, const std::vector<MatrixVectorType> &hamiltonian);

void get_meir_wingreen_k_dependent_current(const Parameters& parameters,
    std::vector<Eigen::MatrixXcd>& green_function,
    std::vector<Eigen::MatrixXcd>& green_function_lesser, const std::vector<Eigen::MatrixXcd>& coupling_left,
    const std::vector<Eigen::MatrixXcd>& coupling_right, const int voltage_step, dcomp* current_left, dcomp* current_right);

void get_embedding_self_energy(Parameters &parameters, const MatrixVectorType &self_energy_left, const MatrixVectorType &self_energy_right, 
	std::vector<dcomp> &self_energy_lesser_i_j, std::vector<dcomp> &self_energy_lesser_j_i, std::vector<dcomp> &retarded_embedding_self_energy_i_j,
    const int voltage_step, const int i, const int j);

double get_k_dependent_bond_current(const Parameters &parameters, const dcomp &density_matrix, const MatrixType &hamiltonian, 
	const std::vector<Eigen::MatrixXcd> &gf_retarded, const std::vector<Eigen::MatrixXcd> &gf_lesser,
	const std::vector<dcomp> &self_energy_lesser_i_j, const std::vector<dcomp> &self_energy_lesser_j_i, const std::vector<dcomp> &retarded_embedding_self_energy_i_j,
	const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const int i, const int j);

void get_bond_current(Parameters &parameters, const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> 
	&self_energy_mb_lesser, const std::vector<std::vector<EmbeddingSelfEnergy>> &leads,	const int voltage_step, double *current,
	 const std::vector<MatrixVectorType> &hamiltonian);

double get_orbital_current(Parameters &parameters, const std::vector<std::vector<dcomp>> &self_energy_mb, 
	const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const std::vector<std::vector<EmbeddingSelfEnergy>> &leads,
	const int voltage_step, const std::vector<MatrixVectorType> &hamiltonian, int i, int j);
