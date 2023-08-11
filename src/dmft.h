#pragma once
#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include "aim.h"
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>


void get_difference(const Parameters &parameters, MatrixVectorType &gf_local_up, MatrixVectorType &old_green_function,
                double &difference, int &index);


void dmft(const Parameters &parameters, const int voltage_step, 
        std::vector<std::vector<dcomp>> &self_energy_mb_up, std::vector<std::vector<dcomp>> &self_energy_mb_down, 
        std::vector<std::vector<dcomp>> &self_energy_mb_lesser_up, std::vector<std::vector<dcomp>> &self_energy_mb_lesser_down,
        std::vector<std::vector<dcomp>> &self_energy_mb_greater_up, std::vector<std::vector<dcomp>> &self_energy_mb_greater_down,
        MatrixVectorType &gf_local_up, MatrixVectorType &gf_local_down,
        MatrixVectorType &gf_local_lesser_up, MatrixVectorType &gf_local_lesser_down,
        MatrixVectorType &gf_local_greater_up, MatrixVectorType &gf_local_greater_down,
        const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<double> &spins_occup, 
        const std::vector<MatrixVectorType> &hamiltonian_up,
	const std::vector<MatrixVectorType> &hamiltonian_down);