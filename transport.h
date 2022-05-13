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
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<dcomp> &transfer_up, std::vector<dcomp> &transmission_down, int voltage_step);

void get_landauer_buttiker_current(Parameters &parameters, std::vector<dcomp> &transmission_up, 
    std::vector<dcomp> &transmission_down, dcomp *current_up, dcomp *current_down, int votlage_step);