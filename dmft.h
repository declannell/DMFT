#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include <iostream>
#include <vector>
#include <x86_64-linux-gnu/mpich/mpi.h>
#include <eigen3/Eigen/Dense>


void get_spin_occupation(Parameters &parameters, std::vector<dcomp> &gf_lesser_up,
        std::vector<dcomp> &gf_lesser_down, double &spin_up, double &spin_down);

void get_difference(Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local_up,
        double &difference);

void impurity_solver(Parameters &parameters, std::vector<dcomp>  &diag_gf_local_up, std::vector<dcomp>  &diag_gf_local_down,
        std::vector<dcomp>  &impurity_self_energy_up, std::vector<dcomp>  &impurity_self_energy_down);

void dmft(Parameters &parameters, int voltage_step, std::vector<double> const &kx, std::vector<double> const &ky, 
        std::vector<std::vector<dcomp>> &self_energy_mb_up, std::vector<std::vector<dcomp>> &self_energy_mb_down, 
        std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &gf_local_down);


