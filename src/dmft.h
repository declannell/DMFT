#pragma once
#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include "AIM.h"
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

void get_spin_occupation(const Parameters &parameters, const std::vector<dcomp> &gf_lesser_up,
                        const std::vector<dcomp> &gf_lesser_down, double *spin_up, double *spin_down);

void get_difference(const Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &old_green_function,
                double &difference, int &index);

void impurity_solver(const Parameters &parameters, const int voltage_step, 
    AIM &aim_up, AIM &aim_down, double *spin_up, double *spin_down);

void dmft(const Parameters &parameters, const int voltage_step, 
        std::vector<std::vector<dcomp>> &self_energy_mb_up, std::vector<std::vector<dcomp>> &self_energy_mb_down, 
        std::vector<std::vector<dcomp>> &self_energy_mb_lesser_up, std::vector<std::vector<dcomp>> &self_energy_mb_lesser_down,
        std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &gf_local_down,
        std::vector<Eigen::MatrixXcd> &gf_local_lesser_up, std::vector<Eigen::MatrixXcd> &gf_local_lesser_down,
        const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<double> &spins_occup, const std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian);


