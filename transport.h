#pragma once
#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include "dmft.h"
#include <iostream>
#include <vector>
#include <x86_64-linux-gnu/mpich/mpi.h>
#include <eigen3/Eigen/Dense>

void get_transmission(Parameters &parameters, std::vector<double> const &kx, std::vector<double> const &ky, std::vector<std::vector<dcomp>> &self_energy_mb, 
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<dcomp> &transmission_up, std::vector<dcomp> &transmission_down, int voltage_step);

void get_landauer_buttiker_current(Parameters &parameters, std::vector<dcomp> &transmission_up, 
    std::vector<dcomp> &transmission_down, dcomp *current_up, dcomp *current_down, int votlage_step);

void get_meir_wingreen_current(const Parameters &parameters, const double kx, const double ky, const std::vector<std::vector<dcomp>> &self_energy_mb_r, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const std::vector<dcomp> &left_lead_se, const std::vector<dcomp> &right_lead_se,
    const int voltage_step, dcomp *current_down);