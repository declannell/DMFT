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
    const int voltage_step, std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian)
{
	int n_x, n_y;

    n_x =  parameters.num_kx_points; //number of k points to take in x direction
    n_y =  parameters.num_ky_points; //number of k points to take in y direction


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
				Eigen::MatrixXcd coupling_left = Eigen::MatrixXd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length);
				Eigen::MatrixXcd coupling_right = Eigen::MatrixXd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length);
				get_coupling(parameters, leads.at(kx_i).at(ky_i).self_energy_left.at(r), 
					leads.at(kx_i).at(ky_i).self_energy_right.at(r), coupling_left, coupling_right, r);
				
				for( int i = 0; i < 2 * parameters.chain_length; i ++){
					for (int p = 0; p < 2 * parameters.chain_length; p++){
						for (int m = 0; m < 2 * parameters.chain_length; m++){
							for (int n = 0; n < 2 * parameters.chain_length; n++){
								transmission_up.at(r) += 1.0 / num_k_points * (coupling_left(i ,m) * gf_interacting_up.interacting_gf.at(r)(m, n)
									* coupling_right(n, p) * std::conj(gf_interacting_up.interacting_gf.at(r)(i, p)));

								transmission_down.at(r) += 1.0 / num_k_points * (coupling_left(i ,m) * gf_interacting_down.interacting_gf.at(r)(m, n)
									* coupling_right(n, p) * std::conj(gf_interacting_down.interacting_gf.at(r)(i, p)));
							}
						}
					}
				}
			}
		}
	}
}

void get_coupling(const Parameters &parameters, const Eigen::MatrixXcd &self_energy_left, const Eigen::MatrixXcd &self_energy_right, 
	Eigen::MatrixXcd &coupling_left, Eigen::MatrixXcd &coupling_right, int r){

        coupling_left(0, 0) = parameters.j1 * (self_energy_left(0, 0) - conj(self_energy_left(0, 0)));
        
        coupling_left(parameters.chain_length, parameters.chain_length) = parameters.j1 * (self_energy_left(1, 1) - conj(self_energy_left(1, 1)));

        coupling_left(parameters.chain_length, 0) = parameters.j1 * 
            (self_energy_left(1, 0) - conj(self_energy_left(0, 1)));

        coupling_left(0, parameters.chain_length) = parameters.j1 * 
            (self_energy_left(0, 1) - conj(self_energy_left(1, 0)));


        coupling_right(parameters.chain_length - 1, parameters.chain_length - 1) = parameters.j1 *
		(self_energy_right(0, 0) - conj(self_energy_right(0, 0)));
        
        coupling_right(2 * parameters.chain_length - 1, 2 * parameters.chain_length - 1) = parameters.j1  * 
            (self_energy_right(1, 1) - conj(self_energy_right(1, 1)));

        coupling_right(2 * parameters.chain_length  - 1, parameters.chain_length - 1) = parameters.j1 *
            (self_energy_right(1, 0) - conj(self_energy_right(0, 1)));

        coupling_right(parameters.chain_length  - 1, 2 * parameters.chain_length - 1) = parameters.j1 *
            (self_energy_right(0, 1) - conj(self_energy_right(1, 0)));

		//std::cout << "The coupling_left is " <<  std::endl;
		//std::cout << coupling_left << std::endl;
		//std::cout << std::endl;
//
		//std::cout << "The coupling_right is " <<  std::endl;
		//std::cout << coupling_right << std::endl;
		//std::cout << std::endl;

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
    dcomp *current_left, dcomp *current_right, const std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian)
{
	int n_x, n_y;

    n_x =  parameters.num_kx_points; //number of k points to take in x direction
    n_y =  parameters.num_ky_points; //number of k points to take in y direction

	std::cout << "The voltage step is " << voltage_step << std::endl;
    double num_k_points = n_x * n_y;
	for (int kx_i = 0; kx_i < n_x; kx_i++) {
		for (int ky_i = 0; ky_i < n_y; ky_i++) {
			Interacting_GF gf_interacting(parameters, self_energy_mb,
			    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
			    voltage_step, hamiltonian.at(kx_i).at(ky_i));

			std::vector<Eigen::MatrixXcd> gf_lesser(parameters.steps,
			    Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));

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
    std::vector<Eigen::MatrixXcd>& green_function_lesser, const std::vector<Eigen::MatrixXcd>& self_energy_left,
    const std::vector<Eigen::MatrixXcd>& self_energy_right, const int voltage_step, dcomp* current_left, dcomp* current_right)
{
	double delta_energy =
	    (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;

	dcomp trace_left = 0.0, trace_right = 0.0;

	std::ofstream integrand_file;
	integrand_file.open(
		    "textfiles/"
		    "integrand_file.txt");

	Eigen::MatrixXcd coupling_left = Eigen::MatrixXd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length);
	Eigen::MatrixXcd coupling_right = Eigen::MatrixXd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length);

	if (parameters.wbl_approx == true) { //we only need to do this once for the wide band limit
		get_coupling(parameters, self_energy_left.at(0), 
			self_energy_right.at(0), coupling_left, coupling_right, 0);
	}



	for (int r = 0; r < parameters.steps; r++) {
		trace_left = 0, trace_right = 0;
		if (parameters.wbl_approx == false) {
			get_coupling(parameters, self_energy_left.at(r), 
				self_energy_right.at(r), coupling_left, coupling_right, r);
		}
		//this formula is from PHYSICAL REVIEW B 72, 125114 2005
		for (int i = 0; i < 2 * parameters.chain_length; i++){
			for (int j = 0; j < 2 * parameters.chain_length; j++){

				trace_left += fermi_function(parameters.energy.at(r) - parameters.voltage_l.at(voltage_step),
		        	parameters) * coupling_left(i, j) * parameters.j1 * (green_function.at(r)(j, i) - conj(green_function.at(r)(i, j)));

				trace_right += fermi_function(parameters.energy.at(r) - parameters.voltage_r.at(voltage_step),
		        	parameters) * coupling_right(i, j) * parameters.j1 * (green_function.at(r)(j, i) - conj(green_function.at(r)(i, j)));
	
				trace_left += parameters.j1 * coupling_left(i, j) * green_function_lesser.at(r)(j, i);
				trace_right += parameters.j1 * coupling_right(i, j) * green_function_lesser.at(r)(j, i);

			}
		}



		integrand_file << parameters.energy.at(r) << "  "
				<< trace_left.real() << "  "
				<< trace_right.real() <<"\n";
				
		*current_left -= delta_energy * trace_left;
		*current_right -= delta_energy * trace_right;
	}

	integrand_file.close();
}
