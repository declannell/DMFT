#pragma once
#include "dmft.h"
#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <mpi.h>
#include <vector>

void get_transmission(
    const Parameters &parameters, const std::vector<double> &kx,
    const std::vector<double> &ky,
    const std::vector<std::vector<dcomp>> &self_energy_mb,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads,
    std::vector<dcomp> &transmission_up, std::vector<dcomp> &transmission_down,
    const int voltage_step, std::vector<std::vector<Eigen::MatrixXd>> &hamiltonian);

void get_landauer_buttiker_current(const Parameters &parameters,
                                   const std::vector<dcomp> &transmission_up,
                                   const std::vector<dcomp> &transmission_down,
                                   dcomp *current_up, dcomp *current_down,
                                   const int votlage_step);

void get_meir_wingreen_current(
    const Parameters &parameters, std::vector<double> const &kx,
    std::vector<double> const &ky,
    const std::vector<std::vector<dcomp>> &self_energy_mb,
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, const int voltage_step,
    dcomp *current_left, dcomp *current_right, const std::vector<std::vector<Eigen::MatrixXd>> &hamiltonian);

void get_meir_wingreen_k_dependent_current(
    const Parameters &parameters, std::vector<Eigen::MatrixXcd> &green_function,
    std::vector<Eigen::MatrixXcd> &green_function_lesser,
    const std::vector<dcomp> &left_lead_se,
    const std::vector<dcomp> &right_lead_se, const int voltage_step,
    dcomp *current_left, dcomp *current_right);