#include "transport.h"
#include <mpi.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include "utilis.h"
#include "dmft.h"
#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"

void get_k_dependent_transmission(const Parameters& parameters, MatrixVectorType& green_function,
    const MatrixVectorType &coupling_left, const MatrixVectorType &coupling_right, std::vector<dcomp> &transmission) {

	double num_k_points = parameters.num_kx_points * parameters.num_ky_points;
	for (int r = 0; r < parameters.steps_myid; r++) {
		transmission.at(r) += 1.0 / num_k_points * (coupling_left.at(r) * green_function.at(r)
				* coupling_right.at(r) * (green_function.at(r)).adjoint()).trace();
	}
}

void get_transmission_gf_local(
    const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_up,
	const std::vector<std::vector<dcomp>> &self_energy_mb_down,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads,
    std::vector<dcomp> &transmission_up, std::vector<dcomp> &transmission_down,
    const int voltage_step, std::vector<MatrixVectorType> &hamiltonian_up,
    std::vector<MatrixVectorType> &hamiltonian_down, MatrixVectorType &gf_local_up, 
    MatrixVectorType &gf_local_lesser_up, MatrixVectorType &gf_local_down, 
    MatrixVectorType &gf_local_lesser_down) {

	MatrixVectorType coupling_left(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
	MatrixVectorType coupling_right(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));


	double num_k_points = parameters.num_kx_points * parameters.num_ky_points;

    MatrixVectorType gf_lesser_up(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length)); 
   	MatrixVectorType gf_lesser_down(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length)); 

	if (parameters.spin_polarised != 1) {// if there is no extenral field and no interaction the system will be spin degenerate. Only do the up spin channel
		//also it is not a half metal. 
		for (int kx_i = 0; kx_i < parameters.num_kx_points; kx_i++) {
			for (int ky_i = 0; ky_i < parameters.num_ky_points; ky_i++) {
				Interacting_GF gf_interacting(parameters, self_energy_mb_up,
				    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
				    voltage_step, hamiltonian_up.at(kx_i).at(ky_i));

    	        get_gf_lesser_non_eq(parameters, gf_interacting.interacting_gf, 
    	            self_energy_mb_up, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
    	            gf_lesser_up, voltage_step);

				get_coupling(parameters, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right, 
					coupling_left, coupling_right);

    	        for(int r = 0; r < parameters.steps_myid; r++){
    	            gf_local_up.at(r) += gf_interacting.interacting_gf.at(r) * (1.0 / num_k_points);
					gf_local_lesser_up.at(r) += gf_lesser_up.at(r) * (1.0 / num_k_points);
				}
				get_k_dependent_transmission(parameters, gf_interacting.interacting_gf, coupling_left, coupling_right, transmission_up);
			}
		}
		for (int r = 0; r < parameters.steps_myid; r++) {
			transmission_down.at(r) = transmission_up.at(r);
		}
	} else { //there is an external magnetic field and therefore we need to consider both spin channels. The system could also be a half metal.
		for (int kx_i = 0; kx_i < parameters.num_kx_points; kx_i++) {
			for (int ky_i = 0; ky_i < parameters.num_ky_points; ky_i++) {
				Interacting_GF gf_interacting_up(parameters, self_energy_mb_up,
				    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
				    voltage_step, hamiltonian_up.at(kx_i).at(ky_i));

				Interacting_GF gf_interacting_down(parameters, self_energy_mb_up,
				    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
				    voltage_step, hamiltonian_down.at(kx_i).at(ky_i));

				get_coupling(parameters, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right, 
					coupling_left, coupling_right);

    	        get_gf_lesser_non_eq(parameters, gf_interacting_up.interacting_gf, 
    	            self_energy_mb_up, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
    	            gf_lesser_up, voltage_step);

    	        get_gf_lesser_non_eq(parameters, gf_interacting_down.interacting_gf, 
    	            self_energy_mb_up, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
    	            gf_lesser_down, voltage_step);

    	        for(int r = 0; r < parameters.steps_myid; r++){
    	            gf_local_up.at(r) += gf_interacting_up.interacting_gf.at(r) * (1.0 / num_k_points);
    	            gf_local_down.at(r) += gf_interacting_down.interacting_gf.at(r) * (1.0 / num_k_points);		  
					gf_local_lesser_up.at(r) += gf_lesser_up.at(r) * (1.0 / num_k_points);
					gf_local_lesser_down.at(r) += gf_lesser_down.at(r) *(1.0 / num_k_points);
				}
				get_k_dependent_transmission(parameters, gf_interacting_up.interacting_gf, coupling_left, coupling_right, transmission_up);
				get_k_dependent_transmission(parameters, gf_interacting_down.interacting_gf, coupling_left, coupling_right, transmission_down);
			}
		}
	}
}

void get_coupling(const Parameters &parameters, const MatrixVectorType &self_energy_left, const MatrixVectorType &self_energy_right, 
	MatrixVectorType &coupling_left, MatrixVectorType &coupling_right){
	for (int r = 0; r < parameters.steps_myid; r++) {
    	coupling_left.at(r) = parameters.j1 * (self_energy_left.at(r) - (self_energy_left.at(r)).adjoint());
    	coupling_right.at(r) = parameters.j1 * (self_energy_right.at(r) - (self_energy_right.at(r)).adjoint());
	}
}

void get_landauer_buttiker_current(const Parameters& parameters,
    const std::vector<dcomp>& transmission_up, const std::vector<dcomp>& transmission_down,
    double* current_up, double* current_down, const int votlage_step)
{
	*current_up = 0;
	*current_down = 0;
	for (int r = 0; r < parameters.steps_myid; r++) {
		*current_up -= (parameters.delta_energy * transmission_up.at(r) * 
		(fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)) + parameters.voltage_l[votlage_step], parameters)
		- fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)) + parameters.voltage_r[votlage_step], parameters))).real();

		*current_down -= (parameters.delta_energy * transmission_down.at(r) *
		(fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)) + parameters.voltage_l[votlage_step], parameters)
		- fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)) + parameters.voltage_r[votlage_step], parameters))).real();	
	}
	
}

void get_meir_wingreen_current(
    const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, const int voltage_step, double *current_left,
	double *current_right, std::vector<dcomp>& transmission_up, const std::vector<MatrixVectorType> &hamiltonian)
{
    int n_x =  parameters.num_kx_points; //number of k points to take in x direction
    int n_y =  parameters.num_ky_points; //number of k points to take in y direction

	MatrixVectorType coupling_left(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
	MatrixVectorType coupling_right(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));

	double num_k_points = n_x * n_y;
	for (int kx_i = 0; kx_i < n_x; kx_i++) {
		for (int ky_i = 0; ky_i < n_y; ky_i++) {
			Interacting_GF gf_interacting(parameters, self_energy_mb,
			    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
			    voltage_step, hamiltonian.at(kx_i).at(ky_i));

			MatrixVectorType gf_lesser(parameters.steps_myid,
			    MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));

			get_gf_lesser_non_eq(parameters, gf_interacting.interacting_gf, self_energy_mb_lesser,
			    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
			    gf_lesser, voltage_step);

			dcomp current_k_resolved_left, current_k_resolved_right;

			get_coupling(parameters, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right, 
				coupling_left, coupling_right);

			get_k_dependent_transmission(parameters, gf_interacting.interacting_gf, coupling_left, coupling_right, transmission_up);

			get_meir_wingreen_k_dependent_current(parameters, gf_interacting.interacting_gf,
			    gf_lesser, coupling_left, coupling_right, voltage_step, &current_k_resolved_left, &current_k_resolved_right);

			*current_left -= (current_k_resolved_left).real() / num_k_points;
			*current_right -= (current_k_resolved_right).real() / num_k_points;
		}
	}
}


void get_meir_wingreen_k_dependent_current(const Parameters& parameters,
    MatrixVectorType& green_function,
    MatrixVectorType& green_function_lesser, const MatrixVectorType& coupling_left,
    const MatrixVectorType& coupling_right, const int voltage_step, dcomp* current_left, dcomp* current_right){
	dcomp trace_left = 0.0, trace_right = 0.0;

	for (int r = 0; r < parameters.steps_myid; r++) {
		trace_left = 0, trace_right = 0;
		int y = r + parameters.start.at(parameters.myid);
		trace_left += (fermi_function(parameters.energy.at(y) - parameters.voltage_l.at(voltage_step),
			parameters) * coupling_left.at(r) * parameters.j1 * (green_function.at(r) - (green_function.at(r)).adjoint())).trace();
		trace_right += (fermi_function(parameters.energy.at(y) - parameters.voltage_r.at(voltage_step),
			parameters) * coupling_right.at(r) * parameters.j1 * (green_function.at(r) - (green_function.at(r)).adjoint())).trace();

		trace_left += (parameters.j1 * coupling_left.at(r) * green_function_lesser.at(r)).trace();
		trace_right += (parameters.j1 * coupling_right.at(r) * green_function_lesser.at(r)).trace();
	}
		*current_left -= parameters.delta_energy * trace_left;
		*current_right -= parameters.delta_energy * trace_right;
}
