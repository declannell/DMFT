#pragma once
#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <eigen3/Eigen/Dense>


void get_spin_occupation(const Parameters &parameters, const std::vector<dcomp> &gf_lesser_up,
                        const std::vector<dcomp> &gf_lesser_down, double *spin_up, double *spin_down);

void get_difference(const Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &old_green_function,
                double &difference, int &index);

void fluctuation_dissipation(const Parameters &parameters, const std::vector<dcomp> &green_function, std::vector<dcomp> &lesser_green_function);

double get_prefactor(const int i, const int j, const int r, const int voltage_step, const Parameters &parameters, const std::vector<std::vector<dcomp>>   &self_energy_mb_up,
    const std::vector<std::vector<dcomp>>   &self_energy_mb_down, const std::vector<std::vector<dcomp>>  &self_energy_mb_lesser_up,
    const std::vector<std::vector<dcomp>>   &self_energy_mb_lesser_down,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads);

double integrate_equilibrium(const Parameters& parameters, const std::vector<double>& gf_1, const std::vector<double>& gf_2, const std::vector<double>& gf_3, const int r,
    const std::vector<std::vector<dcomp>>  &self_energy_mb_up, const std::vector<std::vector<dcomp>>  &self_energy_mb_down,
    const std::vector<std::vector<dcomp>>   &self_energy_mb_lesser_up, const std::vector<std::vector<dcomp>>   &self_energy_mb_lesser_down, int voltage_step,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads);

dcomp integrate(const Parameters& parameters, const std::vector<dcomp>& gf_1, const std::vector<dcomp>& gf_2, const std::vector<dcomp>& gf_3, const int r);

double kramer_kronig_relation(const Parameters& parameters, std::vector<double>& impurity_self_energy_imag, int r);

void self_energy_2nd_order_kramers_kronig(const Parameters& parameters, const std::vector<dcomp>& impurity_gf_up, const std::vector<dcomp>& impurity_gf_down,
    const std::vector<dcomp>& impurity_gf_up_lesser, const std::vector<dcomp>& impurity_gf_down_lesser, std::vector<dcomp>& impurity_self_energy,
    std::vector<dcomp>& impurity_self_energy_lesser_up, const int voltage_step, const std::vector<std::vector<dcomp>>   &self_energy_mb_up, const std::vector<std::vector<dcomp>>   &self_energy_mb_down,
    const std::vector<std::vector<dcomp>>   &self_energy_mb_lesser_up, const std::vector<std::vector<dcomp>>   &self_energy_mb_lesser_down,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads);

void impurity_solver(const Parameters &parameters, const int voltage_step, const std::vector<dcomp>  &impurity_gf_up, const std::vector<dcomp>  &impurity_gf_down,
    const std::vector<dcomp>  &impurity_gf_lesser_up, const std::vector<dcomp>  &impurity_gf_lesser_down,
    std::vector<dcomp>  &impurity_self_energy_up, std::vector<dcomp>  &impurity_self_energy_down, 
    std::vector<dcomp>  &impurity_self_energy_lesser_up, std::vector<dcomp>  &impurity_self_energy_lesser_down,
    double *spin_up, double *spin_down, const std::vector<std::vector<dcomp>> &self_energy_mb_up, const std::vector<std::vector<dcomp>>  &self_energy_mb_down,
    const std::vector<std::vector<dcomp>>  &self_energy_mb_lesser_up, const std::vector<std::vector<dcomp>>  &self_energy_mb_lesser_down,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads);

void dmft(const Parameters &parameters, const int voltage_step, 
        std::vector<std::vector<dcomp>> &self_energy_mb_up, std::vector<std::vector<dcomp>> &self_energy_mb_down, 
        std::vector<std::vector<dcomp>> &self_energy_mb_lesser_up, std::vector<std::vector<dcomp>> &self_energy_mb_lesser_down,
        std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &gf_local_down,
        std::vector<Eigen::MatrixXcd> &gf_local_lesser_up, std::vector<Eigen::MatrixXcd> &gf_local_lesser_down,
        const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<double> &spins_occup, const std::vector<std::vector<Eigen::MatrixXd>> &hamiltonian);


