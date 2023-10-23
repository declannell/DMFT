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

	for (int r = 0; r < parameters.steps_myid; r++) {
		transmission_up.at(r) = 0.0;
	}

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
    std::vector<Eigen::MatrixXcd>& green_function,
    std::vector<Eigen::MatrixXcd>& green_function_lesser, const std::vector<Eigen::MatrixXcd>& coupling_left,
    const std::vector<Eigen::MatrixXcd>& coupling_right, const int voltage_step, dcomp* current_left, dcomp* current_right)
{

	dcomp trace_left = 0.0, trace_right = 0.0;

	for (int r = 0; r < parameters.steps_myid; r++) {
		trace_left = 0, trace_right = 0;
		int y = r + parameters.start.at(parameters.myid);

		MatrixType spec_func = MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);

		spec_func = parameters.j1 * (green_function.at(r) - (green_function.at(r)).adjoint());
		
		trace_left += (fermi_function(parameters.energy.at(y) - parameters.voltage_l.at(voltage_step),
	    	parameters) * coupling_left.at(r) * spec_func).trace();
		trace_right += (fermi_function(parameters.energy.at(y) - parameters.voltage_r.at(voltage_step),
	    	parameters) * coupling_right.at(r) * spec_func).trace();
		
		trace_left += (parameters.j1 * coupling_left.at(r) * green_function_lesser.at(r)).trace();
		trace_right += (parameters.j1 * coupling_right.at(r) * green_function_lesser.at(r)).trace();
		
		//if (parameters.myid == 0) std::cout << (-2.0 * green_function.at(r)(0, 0)).imag() << " " << (green_function_lesser.at(r)(0,0)).imag() << " " << voltage_step << std::endl;

		*current_left -= parameters.delta_energy * trace_left;
		*current_right -= parameters.delta_energy * trace_right;
	}
}

void get_embedding_self_energy(Parameters &parameters, const MatrixVectorType &self_energy_left, const MatrixVectorType &self_energy_right, 
	std::vector<dcomp> &self_energy_lesser_i_j, std::vector<dcomp> &self_energy_lesser_j_i, std::vector<dcomp> &retarded_embedding_self_energy_i_j,
    const int voltage_step, const int i, const int j) {

	for (int r = 0; r < parameters.steps_myid; r++) {
		retarded_embedding_self_energy_i_j.at(r) = self_energy_left.at(r)(i, j) + self_energy_right.at(r)(i, j);

		self_energy_lesser_i_j.at(r) = - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) * 
        (self_energy_left.at(r)(i, j) - std::conj(self_energy_left.at(r)(j, i))) 
		- fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters) * 
        (self_energy_right.at(r)(i, j) - std::conj(self_energy_right.at(r)(j, i)));

		self_energy_lesser_j_i.at(r) = - std::conj(self_energy_lesser_i_j.at(r));
	}
}

double get_k_dependent_bond_current(const Parameters &parameters, const dcomp &density_matrix, const MatrixType &hamiltonian, 
	const std::vector<Eigen::MatrixXcd> &gf_retarded, const std::vector<Eigen::MatrixXcd> &gf_lesser,
	const std::vector<dcomp> &self_energy_lesser_i_j, const std::vector<dcomp> &self_energy_lesser_j_i, const std::vector<dcomp> &retarded_embedding_self_energy_i_j,
	const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const int i, const int j){
		dcomp current_k_myid = 0, current_k = 0;
		for (int r = 0; r < parameters.steps_myid; r++) {
			current_k_myid += gf_retarded.at(r)(i, j) * self_energy_lesser_j_i.at(r) + gf_lesser.at(r)(i, j) * std::conj(retarded_embedding_self_energy_i_j.at(r)) 
				- self_energy_lesser_i_j.at(r) * std::conj(gf_retarded.at(r)(i, j)) - retarded_embedding_self_energy_i_j.at(r) * gf_lesser.at(r)(j, i);

			if (i == 0 && j == 1) std::cout << "gf_retarded.at(r)(i, j) = " << gf_retarded.at(r)(i, j) <<  " self_energy_lesser_j_i.at(r) = "
				<< self_energy_lesser_j_i.at(r) << std::endl;
		}
		current_k_myid = parameters.delta_energy * current_k / (2.0 * M_PI);
		MPI_Allreduce(&current_k_myid, &current_k, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
		current_k += 2.0 * (hamiltonian(i, j) * density_matrix).imag();
		//std::cout << current_k << "  " <<  (density_matrix) << " " << i << " " << j << std::endl;
		return current_k.real();
}

void get_bond_current(Parameters &parameters, const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> 
	&self_energy_mb_lesser, const std::vector<std::vector<EmbeddingSelfEnergy>> &leads,	const int voltage_step, double *current,
	 const std::vector<MatrixVectorType> &hamiltonian) {
	//this is setting up whether the orbital is to the left or right of the interface. 
	std::vector<int> left;
	std::vector<int> right;


	for (int i = 0; i < parameters.chain_length; i++) {		
		if (i < parameters.interface) {
			left.push_back(i);
			left.push_back(i + parameters.chain_length);
			left.push_back(i + 2 * parameters.chain_length);
			left.push_back(i + 3 * parameters.chain_length);
		} else if (parameters.interface <= i) {
			right.push_back(i);
			right.push_back(i + parameters.chain_length);
			right.push_back(i + 2 * parameters.chain_length);
			right.push_back(i + 3 * parameters.chain_length);
		}
	}

	if (parameters.myid == 0) std::cout << left.size() << " is the number of orbitals on the left \n" << std::endl;
	if (parameters.myid == 0) std::cout << right.size() << " is the number of orbitals on the right \n" << std::endl;


	//this is implementing eq 32 of Ivan's, andrea and maria bond current approach.
	for (std::vector<int>::size_type i = 0; i < left.size(); i++) {
		for (std::vector<int>::size_type j = 0; j < right.size(); j++) {
			double orbital_current = get_orbital_current(parameters, self_energy_mb, self_energy_mb_lesser, leads, voltage_step, hamiltonian, left.at(i), right.at(j));
			//if (parameters.myid == 0) std::cout << orbital_current << "(" << left.at(i) << "," << right.at(j) << ")" << std::endl;
			*current += orbital_current;
		}
	}
	//if (parameters.myid == 0) std::cout << "the total bond current " << *current << std::endl;
}

double get_orbital_current(Parameters &parameters, const std::vector<std::vector<dcomp>> &self_energy_mb, 
	const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const std::vector<std::vector<EmbeddingSelfEnergy>> &leads,
	const int voltage_step, const std::vector<MatrixVectorType> &hamiltonian, int i, int j) {

    int n_x =  parameters.num_kx_points; //number of k points to take in x direction
    int n_y =  parameters.num_ky_points; //number of k points to take in y direction

	//MatrixVectorType coupling_left(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
	//MatrixVectorType coupling_right(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
	std::vector<dcomp> self_energy_lesser_i_j(parameters.steps, 0), self_energy_lesser_j_i(parameters.steps, 0), 
		retarded_embedding_self_energy_i_j(parameters.steps, 0);
	dcomp density_matrix_k;
	double current_k = 0;

	double num_k_points = n_x * n_y;
	for (int kx_i = 0; kx_i < n_x; kx_i++) {
		for (int ky_i = 0; ky_i < n_y; ky_i++) {
			Interacting_GF gf_interacting(parameters, self_energy_mb,
			    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
			    voltage_step, hamiltonian.at(kx_i).at(ky_i));

			MatrixVectorType gf_lesser(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));

			get_gf_lesser_non_eq(parameters, gf_interacting.interacting_gf, self_energy_mb_lesser,
			    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
			    gf_lesser, voltage_step);

			get_density_matrix(parameters, gf_lesser, density_matrix_k, j, i);
			//std::cout << "the density matrix element is " << density_matrix_k << " " << j << " " << i << std::endl;

			get_embedding_self_energy(parameters, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right, 
				self_energy_lesser_i_j, self_energy_lesser_j_i, retarded_embedding_self_energy_i_j, voltage_step, i, j);

			current_k += get_k_dependent_bond_current(parameters, density_matrix_k, hamiltonian.at(kx_i).at(ky_i), gf_interacting.interacting_gf, gf_lesser, 
				self_energy_lesser_i_j, self_energy_lesser_j_i, retarded_embedding_self_energy_i_j, self_energy_mb, self_energy_mb_lesser, i , j);
			
		}
	}

	return current_k * 2.0 * M_PI / num_k_points;
}