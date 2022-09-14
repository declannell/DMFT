#include "transport.h"

#include <mpi.h>

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

#include "dmft.h"
#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"

void get_transmission(
    const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_up,
	const std::vector<std::vector<dcomp>> &self_energy_mb_down,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads,
    std::vector<dcomp> &transmission_up, std::vector<dcomp> &transmission_down,
    const int voltage_step, std::vector<std::vector<Eigen::MatrixXd>> &hamiltonian)
{
	int n_x, n_y;
    if (parameters.leads_3d == false){
        n_x =  parameters.num_kx_points; //number of k points to take in x direction
        n_y =  parameters.num_ky_points; //number of k points to take in y direction
    } else {
        n_x = 1;
        n_y = 1;
    }

    double num_k_points = n_x * n_y;


	for (int kx_i = 0; kx_i < n_x; kx_i++) {
		for (int ky_i = 0; ky_i < n_y; ky_i++) {
			Interacting_GF gf_interacting_up(parameters, self_energy_mb_up,
			    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
			    voltage_step, hamiltonian.at(kx_i).at(ky_i));

			Interacting_GF gf_interacting_down(parameters, self_energy_mb_down,
			    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
			    voltage_step, hamiltonian.at(kx_i).at(ky_i));

			for (int r = 0; r < parameters.steps; r++) {
				dcomp coupling_left = 2 * leads.at(kx_i).at(ky_i).self_energy_left.at(r).imag();
				dcomp coupling_right = 2 * leads.at(kx_i).at(ky_i).self_energy_right.at(r).imag();
				transmission_up.at(r) += 1.0 / num_k_points
				    * (coupling_left
				        * gf_interacting_up.interacting_gf.at(r)(0, parameters.chain_length - 1)
				        * coupling_right
				        * std::conj(gf_interacting_up.interacting_gf.at(
				            r)(0, parameters.chain_length - 1)));
				transmission_down.at(r) += 1.0 / num_k_points
				    * (coupling_left
				        * gf_interacting_down.interacting_gf.at(r)(0, parameters.chain_length - 1)
				        * coupling_right
				        * std::conj(gf_interacting_up.interacting_gf.at(
				            r)(0, parameters.chain_length - 1)));
			}
		}
	}
}

void get_landauer_buttiker_current(const Parameters& parameters,
    const std::vector<dcomp>& transmission_up, const std::vector<dcomp>& transmission_down,
    dcomp* current_up, dcomp* current_down, const int votlage_step)
{
	double delta_energy =
	    (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
	for (int r = 0; r < parameters.steps; r++) {
		*current_up -= delta_energy * transmission_up.at(r)
		    * (fermi_function(parameters.energy.at(r) + parameters.voltage_l[votlage_step],
		           parameters)
		        - fermi_function(parameters.energy.at(r)
		                + parameters.voltage_r[votlage_step],
		            parameters));
		*current_down -= delta_energy * transmission_down.at(r)
		    * (fermi_function(parameters.energy.at(r) + parameters.voltage_l[votlage_step],
		           parameters)
		        - fermi_function(parameters.energy.at(r)
		                + parameters.voltage_r[votlage_step],
		            parameters));
	}
}

void get_meir_wingreen_current(
    const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb,
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, const int voltage_step,
    dcomp *current_left, dcomp *current_right, const std::vector<std::vector<Eigen::MatrixXd>> &hamiltonian)
{
	int n_x, n_y;
    if (parameters.leads_3d == false){
        n_x =  parameters.num_kx_points; //number of k points to take in x direction
        n_y =  parameters.num_ky_points; //number of k points to take in y direction
    } else {
        n_x = 1;
        n_y = 1;
    }

    double num_k_points = n_x * n_y;
	for (int kx_i = 0; kx_i < n_x; kx_i++) {
		for (int ky_i = 0; ky_i < n_y; ky_i++) {
			Interacting_GF gf_interacting(parameters, self_energy_mb,
			    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
			    voltage_step, hamiltonian.at(kx_i).at(ky_i));

			std::vector<Eigen::MatrixXcd> gf_lesser(parameters.steps,
			    Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
			get_gf_lesser_non_eq(parameters, gf_interacting.interacting_gf, self_energy_mb_lesser,
			    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
			    gf_lesser, voltage_step);

			dcomp current_k_resolved_left, current_k_resolved_right;

			get_meir_wingreen_k_dependent_current(parameters, gf_interacting.interacting_gf,
			    gf_lesser, leads.at(kx_i).at(ky_i).self_energy_left,
			    leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, &current_k_resolved_left, &current_k_resolved_right);

			*current_left -= current_k_resolved_left / num_k_points;
			*current_right -= current_k_resolved_right / num_k_points;
		}
	}
}

void get_meir_wingreen_k_dependent_current(const Parameters& parameters,
    std::vector<Eigen::MatrixXcd>& green_function,
    std::vector<Eigen::MatrixXcd>& green_function_lesser, const std::vector<dcomp>& left_lead_se,
    const std::vector<dcomp>& right_lead_se, const int voltage_step, dcomp* current_left, dcomp* current_right)
{
	double delta_energy =
	    (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
	dcomp trace_left, trace_right, coupling_left, coupling_right, spectral_left, spectral_right;
    //std::ofstream integrand_file;
    //integrand_file.open(
    //    "textfiles/"
    //    "integrand.txt");

	for (int r = 0; r < parameters.steps; r++) {
		coupling_left = parameters.j1 * (left_lead_se.at(r) - std::conj(left_lead_se.at(r)));
		coupling_right = parameters.j1 * (right_lead_se.at(r) - std::conj(right_lead_se.at(r)));
		spectral_left =
		    parameters.j1 * (green_function.at(r)(0, 0) - std::conj(green_function.at(r)(0, 0)));
		spectral_right = parameters.j1
		    * (green_function.at(r)(parameters.chain_length - 1, parameters.chain_length - 1)
		        - std::conj(green_function.at(
		            r)(parameters.chain_length - 1, parameters.chain_length - 1)));

		dcomp trace_left_a =
		    fermi_function(parameters.energy.at(r) - parameters.voltage_l.at(voltage_step),
		        parameters)
		    * coupling_left * spectral_left;
		dcomp trace_right_a =
		    fermi_function(parameters.energy.at(r) - parameters.voltage_r.at(voltage_step),
		        parameters)
		    * coupling_right * spectral_right;
		dcomp trace_left_b = parameters.j1 * coupling_left * green_function_lesser.at(r)(0, 0);
		dcomp trace_right_b = parameters.j1 * coupling_right
		    * green_function_lesser.at(r)(parameters.chain_length - 1, parameters.chain_length - 1);

		trace_left = trace_left_a + trace_left_b;
		trace_right = trace_right_a + trace_right_b;		

		//integrand_file << parameters.energy.at(r) << "  "
		//		<< trace_left_a.real() << "  "
		//		<< trace_right_a.real() << "  "
		//		<< trace_left_b.real() << "  "
		//		<< trace_right_b.real() << "  "
		//		<< trace_left.real() << "  "
		//		<< trace_right.real() <<"\n";
				
		*current_left -= delta_energy * trace_left;
		*current_right -= delta_energy * trace_right;
	}

	//integrand_file.close();
}
