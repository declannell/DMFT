#pragma once
#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include "dmft.h"
#include <iostream>
#include <vector>
#include <x86_64-linux-gnu/mpich/mpi.h>
#include <eigen3/Eigen/Dense>

void get_transmission(const Parameters &parameters, const std::vector<double> &kx, const std::vector<double> &ky, const std::vector<std::vector<dcomp>> &self_energy_mb, 
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<dcomp> &transmission_up, std::vector<dcomp> &transmission_down, const int voltage_step);

void get_landauer_buttiker_current(const Parameters &parameters, const std::vector<dcomp> &transmission_up, 
    const std::vector<dcomp> &transmission_down, dcomp *current_up, dcomp *current_down, const int votlage_step);

void get_meir_wingreen_current(Parameters &parameters, std::vector<double> const &kx, std::vector<double> const &ky, std::vector<std::vector<dcomp>> &self_energy_mb, 
    std::vector<std::vector<dcomp>> &self_energy_mb_lesser, std::vector<std::vector<EmbeddingSelfEnergy>> &leads, int voltage_step, dcomp *current);

void get_meir_wingreen_k_dependent_current(const Parameters &parameters, std::vector<Eigen::MatrixXcd> &green_function, std::vector<Eigen::MatrixXcd> &green_function_lesser,
    const std::vector<dcomp> &left_lead_se, const std::vector<dcomp> &right_lead_se, const int voltage_step, dcomp *current);