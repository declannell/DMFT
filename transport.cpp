#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include "dmft.h"
#include "transport.h"
#include <iostream>
#include <vector>
#include <x86_64-linux-gnu/mpich/mpi.h>
#include <eigen3/Eigen/Dense>

void get_transmission(const Parameters &parameters, const std::vector<double> &kx, const std::vector<double> &ky, const std::vector<std::vector<dcomp>> &self_energy_mb, 
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<dcomp> &transmission_up, std::vector<dcomp> &transmission_down, const int voltage_step){

    double num_k_points = parameters.chain_length_x * parameters.chain_length_y;
    std::cout << num_k_points << std::endl;

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
                dcomp coupling_left = 2 * leads.at(kx_i).at(ky_i).self_energy_left.at(r).imag();
                dcomp coupling_right = 2* leads.at(kx_i).at(ky_i).self_energy_right.at(r).imag();
                transmission_up.at(r) += 1.0 / num_k_points * (coupling_left * gf_interacting_up.interacting_gf.at(r)(0, parameters.chain_length - 1) *
                    coupling_right * std::conj(gf_interacting_up.interacting_gf.at(r)(0, parameters.chain_length - 1)));
                transmission_down.at(r) += 1.0 / num_k_points * (coupling_left * gf_interacting_down.interacting_gf.at(r)(0, parameters.chain_length - 1) *
                    coupling_right * std::conj(gf_interacting_up.interacting_gf.at(r)(0, parameters.chain_length - 1))); 
            }
        }
    }
}

void get_landauer_buttiker_current(const Parameters &parameters, const std::vector<dcomp> &transmission_up, 
    const std::vector<dcomp> &transmission_down, dcomp *current_up, dcomp *current_down, const int votlage_step){
    double delta_energy = (parameters.e_upper_bound -
                parameters.e_lower_bound) / (double)parameters.steps;
    for(int r = 0; r < parameters.steps; r++){
        *current_up -= delta_energy * transmission_up.at(r) * (fermi_function(parameters.energy.at(r).real() + parameters.voltage_l[votlage_step], parameters)
        - fermi_function(parameters.energy.at(r).real() + parameters.voltage_r[votlage_step], parameters)); 
        *current_down -= delta_energy * transmission_down.at(r) * (fermi_function(parameters.energy.at(r).real() + parameters.voltage_l[votlage_step], parameters)
        - fermi_function(parameters.energy.at(r).real() + parameters.voltage_r[votlage_step], parameters)); 
    }
}

void get_meir_wingreen_current(Parameters &parameters, std::vector<double> const &kx, std::vector<double> const &ky, std::vector<std::vector<dcomp>> &self_energy_mb, 
    std::vector<std::vector<dcomp>> &self_energy_mb_lesser, std::vector<std::vector<EmbeddingSelfEnergy>> &leads, int voltage_step, dcomp *current){

    double num_k_points = parameters.chain_length_x * parameters.chain_length_y;
    for(int kx_i = 0; kx_i < parameters.chain_length_x; kx_i++) {
        for(int ky_i = 0; ky_i < parameters.chain_length_y; ky_i++) {
            Interacting_GF gf_interacting(parameters,
                kx.at(kx_i), ky.at(ky_i), self_energy_mb,
                leads.at(kx_i).at(ky_i).self_energy_left,
                leads.at(kx_i).at(ky_i).self_energy_right, voltage_step);

            std::vector<Eigen::MatrixXcd> gf_lesser(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
            get_gf_lesser_non_eq(parameters, gf_interacting.interacting_gf, self_energy_mb_lesser, leads.at(kx_i).at(ky_i).self_energy_left,
            leads.at(kx_i).at(ky_i).self_energy_right, gf_lesser, voltage_step);

            dcomp current_k_resolved;       

            get_meir_wingreen_k_dependent_current(parameters, gf_interacting.interacting_gf, gf_lesser,
            leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, &current_k_resolved);

            *current -= current_k_resolved / num_k_points;
        }
    }
}


void get_meir_wingreen_k_dependent_current(const Parameters &parameters, std::vector<Eigen::MatrixXcd> &green_function, std::vector<Eigen::MatrixXcd> &green_function_lesser,
    const std::vector<dcomp> &left_lead_se, const std::vector<dcomp> &right_lead_se, const int voltage_step, dcomp *current){

    double delta_energy = (parameters.e_upper_bound -
                parameters.e_lower_bound) / (double)parameters.steps;
    dcomp trace, coupling_left, coupling_right, spectral_left, spectral_right;

    for(int r = 0; r < parameters.steps; r++){
        coupling_left = parameters.j1 * (left_lead_se.at(r) - std::conj(left_lead_se.at(r)));
        coupling_right = parameters.j1 * (right_lead_se.at(r) - std::conj(right_lead_se.at(r)));
        spectral_left = parameters.j1 * (green_function.at(r)(0, 0) - std::conj(green_function.at(r)(0, 0)));
        spectral_right = parameters.j1 * (green_function.at(r)(parameters.chain_length - 1, parameters.chain_length - 1) 
            - std::conj(green_function.at(r)(parameters.chain_length - 1, parameters.chain_length - 1)));

        trace = fermi_function(parameters.energy.at(r).real() - parameters.voltage_l.at(voltage_step), parameters) * coupling_left * spectral_left;
        trace += - fermi_function(parameters.energy.at(r).real() - parameters.voltage_r.at(voltage_step), parameters) * coupling_right * spectral_right;          
        trace += parameters.j1 * coupling_left * green_function_lesser.at(r)(0, 0);
        trace += - parameters.j1 * coupling_right * green_function_lesser.at(r)(parameters.chain_length - 1, parameters.chain_length - 1);


        *current -= delta_energy * trace * 0.5;
    }
}




