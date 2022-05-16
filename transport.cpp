#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include "dmft.h"
#include "transport.h"
#include <iostream>
#include <vector>
#include <x86_64-linux-gnu/mpich/mpi.h>
#include <eigen3/Eigen/Dense>

void get_transmission(Parameters &parameters, std::vector<double> const &kx, std::vector<double> const &ky, std::vector<std::vector<dcomp>> &self_energy_mb, 
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<dcomp> &transmission_up, std::vector<dcomp> &transmission_down, int voltage_step){

    double num_k_points = parameters.chain_length_x * parameters.chain_length_y;
    for(int kx_i = 0; kx_i < parameters.chain_length_x; kx_i++) {
        for(int ky_i = 0; ky_i < parameters.chain_length_y; ky_i++) {
            Interacting_GF gf_interacting_up(parameters,
                kx.at(kx_i), ky.at(ky_i), self_energy_mb,
                leads.at(kx_i).at(ky_i).self_energy_left,
                leads.at(kx_i).at(ky_i).self_energy_right, voltage_step);

            Interacting_GF gf_interacting_down(parameters,
                kx.at(kx_i), ky.at(ky_i), self_energy_mb,
                leads.at(kx_i).at(ky_i).self_energy_left,
                leads.at(kx_i).at(ky_i).self_energy_right, voltage_step);             

            for(int r = 0; r < parameters.steps; r++){
                dcomp coupling_left = leads.at(kx_i).at(ky_i).self_energy_left.at(r).imag();
                dcomp coupling_right = leads.at(kx_i).at(ky_i).self_energy_right.at(r).imag();
                transmission_up.at(r) += 1.0 / num_k_points * (coupling_left * gf_interacting_up.interacting_gf.at(r)(0, parameters.chain_length - 1) *
                    coupling_right * std::conj(gf_interacting_up.interacting_gf.at(r)(0, parameters.chain_length - 1)));
                transmission_down.at(r) += 1.0 / num_k_points * (coupling_left * gf_interacting_down.interacting_gf.at(r)(0, parameters.chain_length - 1) *
                    coupling_right * std::conj(gf_interacting_up.interacting_gf.at(r)(0, parameters.chain_length - 1))); 
            }
        }
    }
}

void get_landauer_buttiker_current(Parameters &parameters, std::vector<dcomp> &transmission_up, 
    std::vector<dcomp> &transmission_down, dcomp *current_up, dcomp *current_down, int votlage_step){
    double delta_energy = (parameters.e_upper_bound -
                parameters.e_lower_bound) / (double)parameters.steps;
    for(int r = 0; r < parameters.steps; r++){
        *current_up -= delta_energy * transmission_up.at(r) * (fermi_function(parameters.energy.at(r).real() + parameters.voltage_l[votlage_step], parameters)
        - fermi_function(parameters.energy.at(r).real() + parameters.voltage_r[votlage_step], parameters)); 
        *current_down -= delta_energy * transmission_down.at(r) * (fermi_function(parameters.energy.at(r).real() + parameters.voltage_l[votlage_step], parameters)
        - fermi_function(parameters.energy.at(r).real() + parameters.voltage_r[votlage_step], parameters)); 
    }
}


void get_meir_wingreen_current(const Parameters &parameters, const double kx, const double ky, const std::vector<std::vector<dcomp>> &self_energy_mb_r, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const std::vector<dcomp> &left_lead_se, const std::vector<dcomp> &right_lead_se,
    const int voltage_step, dcomp *current_down){

    }




