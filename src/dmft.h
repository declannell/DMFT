#pragma once
#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <eigen3/Eigen/Dense>


void get_spin_occupation(Parameters &parameters, std::vector<dcomp> &gf_lesser_up,
                        std::vector<dcomp> &gf_lesser_down, double *spin_up, double *spin_down);

void get_difference(Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &old_green_function,
                double &difference, int &index);

void fluctuation_dissipation(Parameters &parameters, const std::vector<dcomp> &green_function, std::vector<dcomp> &lesser_green_function);

dcomp integrate(Parameters &parameters, std::vector<dcomp> &gf_1, std::vector<dcomp> &gf_2,
            std::vector<dcomp> &gf_3, int r);

void self_energy_2nd_order(Parameters &parameters, std::vector<dcomp> &impurity_gf_up, std::vector<dcomp> &impurity_gf_down, 
        std::vector<dcomp> &impurity_gf_up_lesser, std::vector<dcomp> &impurity_gf_down_lesser, std::vector<dcomp> &impurity_self_energy,
        std::vector<dcomp> &impurity_self_energy_lesser_up);

void impurity_solver(Parameters &parameters, std::vector<dcomp>  &impurity_gf_up, std::vector<dcomp>  &impurity_gf_down,
    std::vector<dcomp>  &impurity_gf_lesser_up, std::vector<dcomp>  &impurity_gf_lesser_down,
    std::vector<dcomp>  &impurity_self_energy_up, std::vector<dcomp>  &impurity_self_energy_down, 
    std::vector<dcomp>  &impurity_self_energy_lesser_up, std::vector<dcomp>  &impurity_self_energy_lesser_down,
    double *spin_up, double *spin_down);

void dmft(Parameters &parameters, int voltage_step, std::vector<double> const &kx, std::vector<double> const &ky, 
        std::vector<std::vector<dcomp>> &self_energy_mb_up, std::vector<std::vector<dcomp>> &self_energy_mb_down, 
        std::vector<std::vector<dcomp>> &self_energy_mb_lesser_up, std::vector<std::vector<dcomp>> &self_energy_mb_lesser_down,
        std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &gf_local_down,
        std::vector<Eigen::MatrixXcd> &gf_local_lesser_up, std::vector<Eigen::MatrixXcd> &gf_local_lesser_down,
        std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<double> &spins_occup, const std::vector<std::vector<Eigen::MatrixXd>> &hamiltonian);


