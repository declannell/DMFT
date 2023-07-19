#pragma once
#include "dmft.h"
#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

void get_transmission(
    const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_up,
	const std::vector<std::vector<dcomp>> &self_energy_mb_down,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads,
    std::vector<dcomp> &transmission_up, std::vector<dcomp> &transmission_down,
    const int voltage_step, std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian);

void get_transmission_gf_local(
    const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_up,
	const std::vector<std::vector<dcomp>> &self_energy_mb_down,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads,
    std::vector<dcomp> &transmission_up, std::vector<dcomp> &transmission_down,
    const int voltage_step, std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian_up,
    std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian_down, std::vector<Eigen::MatrixXcd> &gf_local_up, 
    std::vector<Eigen::MatrixXcd> &gf_local_lesser_up, std::vector<Eigen::MatrixXcd> &gf_local_down, 
    std::vector<Eigen::MatrixXcd> &gf_local_lesser_down);

void get_coupling(const Parameters &parameters, const Eigen::MatrixXcd &self_energy_left, const Eigen::MatrixXcd &self_energy_right, 
	Eigen::MatrixXcd &coupling_left, Eigen::MatrixXcd &coupling_right, int r);

void get_landauer_buttiker_current(const Parameters &parameters, const std::vector<dcomp> &transmission_up, const std::vector<dcomp> &transmission_down,
    double *current_up, double *current_down, const int votlage_step);

void get_meir_wingreen_current(
    const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, const int voltage_step, double *current_left,
	double *current_right, std::vector<dcomp>& transmission_up, const std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian);
    
void get_meir_wingreen_k_dependent_current(const Parameters& parameters,
    std::vector<Eigen::MatrixXcd>& green_function,
    std::vector<Eigen::MatrixXcd>& green_function_lesser, const std::vector<Eigen::MatrixXcd>& self_energy_left,
    const std::vector<Eigen::MatrixXcd>& self_energy_right, const int voltage_step, dcomp* current_left, dcomp* current_right);

void get_k_dependent_transmission(const Parameters& parameters, std::vector<Eigen::MatrixXcd>& green_function,
    const Eigen::MatrixXcd &coupling_left, Eigen::MatrixXcd &coupling_right, std::vector<dcomp> &transmission);