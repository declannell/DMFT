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

    int n_x =  parameters.num_kx_points; //number of k points to take in x direction
    int n_y =  parameters.num_ky_points; //number of k points to take in y direction
	Eigen::MatrixXcd coupling_left = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
	Eigen::MatrixXcd coupling_right = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);

	if (parameters.wbl_approx == true) {
		get_coupling(parameters, leads.at(0).at(0).self_energy_left.at(0), 
		leads.at(0).at(0).self_energy_right.at(0), coupling_left, coupling_right, 0);
	}    


	if (parameters.spin_polarised == true) {
		for (int kx_i = 0; kx_i < n_x; kx_i++) {
			for (int ky_i = 0; ky_i < n_y; ky_i++) {
				Interacting_GF gf_interacting_up(parameters, self_energy_mb_up,
				    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
				    voltage_step, hamiltonian.at(kx_i).at(ky_i));

				Interacting_GF gf_interacting_down(parameters, self_energy_mb_down,
				    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
				    voltage_step, hamiltonian.at(kx_i).at(ky_i));

				get_k_dependent_transmission(parameters, gf_interacting_up.interacting_gf, coupling_left, coupling_right, transmission_up);
				get_k_dependent_transmission(parameters, gf_interacting_down.interacting_gf, coupling_left, coupling_right, transmission_down);
			}
		}
	} else {//spin_up = spin_down
		for (int kx_i = 0; kx_i < n_x; kx_i++) {
			for (int ky_i = 0; ky_i < n_y; ky_i++) {
				Interacting_GF gf_interacting_up(parameters, self_energy_mb_up,
				    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
				    voltage_step, hamiltonian.at(kx_i).at(ky_i));

				get_k_dependent_transmission(parameters, gf_interacting_up.interacting_gf, coupling_left, coupling_right, transmission_up);
			}
		}
		for (int r = 0; r < parameters.steps_myid; r++) {
			transmission_down.at(r) = transmission_up.at(r);
		}
	}
}

void get_k_dependent_transmission(const Parameters& parameters, std::vector<Eigen::MatrixXcd>& green_function,
    const Eigen::MatrixXcd &coupling_left, Eigen::MatrixXcd &coupling_right, std::vector<dcomp> &transmission) {
		double num_k_points = parameters.num_kx_points * parameters.num_ky_points;
		for (int r = 0; r < parameters.steps_myid; r++) {
			for( int i = 0; i < 4 * parameters.chain_length; i ++){
				for (int n = 0; n < 4 * parameters.chain_length; n++){
					transmission.at(r) += 1.0 / num_k_points * (coupling_left(i ,i) * green_function.at(r)(i, n)
					* coupling_right(n, n) * std::conj(green_function.at(r)(i, n)));
				}
			}
		}
}

void get_transmission_gf_local(
    const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_up,
	const std::vector<std::vector<dcomp>> &self_energy_mb_down,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads,
    std::vector<dcomp> &transmission_up, std::vector<dcomp> &transmission_down,
    const int voltage_step, std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian, std::vector<Eigen::MatrixXcd> &gf_local, 
    std::vector<Eigen::MatrixXcd> &gf_local_lesser) {

	Eigen::MatrixXcd coupling_left = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
	Eigen::MatrixXcd coupling_right = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
	if (parameters.wbl_approx == true) {
		get_coupling(parameters, leads.at(0).at(0).self_energy_left.at(0), 
		leads.at(0).at(0).self_energy_right.at(0), coupling_left, coupling_right, 0);
	}    

	double num_k_points = parameters.num_kx_points * parameters.num_ky_points;

    for(int r = 0; r < parameters.steps_myid; r++){
        gf_local.at(r) = (Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
        gf_local_lesser.at(r) = (Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
    }
    std::vector<Eigen::MatrixXcd> gf_lesser(parameters.steps_myid, Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length)); 

	for (int kx_i = 0; kx_i < parameters.num_kx_points; kx_i++) {
		for (int ky_i = 0; ky_i < parameters.num_ky_points; ky_i++) {
			Interacting_GF gf_interacting(parameters, self_energy_mb_up,
			    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
			    voltage_step, hamiltonian.at(kx_i).at(ky_i));


            get_gf_lesser_non_eq(parameters, gf_interacting.interacting_gf, 
                self_energy_mb_up, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
                gf_lesser, voltage_step);

            for(int r = 0; r < parameters.steps_myid; r++){
                for(int i = 0; i < 4 * parameters.chain_length; i++){
                    for(int j = 0; j < 4 * parameters.chain_length; j++){
                        gf_local.at(r)(i, j) += gf_interacting.interacting_gf.at(r)(i, j) / num_k_points;
                    }
					gf_local_lesser.at(r)(i, i) += gf_lesser.at(r)(i, i) / num_k_points;
                }
			}
			get_k_dependent_transmission(parameters, gf_interacting.interacting_gf, coupling_left, coupling_right, transmission_up);
		}
	}
	for (int r = 0; r < parameters.steps_myid; r++) {
		transmission_down.at(r) = transmission_up.at(r);
	}
}


void get_coupling(const Parameters &parameters, const Eigen::MatrixXcd &self_energy_left, const Eigen::MatrixXcd &self_energy_right, 
	Eigen::MatrixXcd &coupling_left, Eigen::MatrixXcd &coupling_right, int r){

		if (parameters.wbl_approx != true) {
			std::cout << "think about how the coupling matirx should be done if the wba isn't being used \n";
			exit(1);
		}

        coupling_left(0, 0) = parameters.j1 * (self_energy_left(0, 0) - conj(self_energy_left(0, 0)));
        
        coupling_left(parameters.chain_length, parameters.chain_length) = parameters.j1 * (self_energy_left(1, 1) - conj(self_energy_left(1, 1)));

        coupling_left(2 * parameters.chain_length, 2 * parameters.chain_length) = parameters.j1 * (self_energy_left(2, 2) - conj(self_energy_left(2, 2)));

        coupling_left(3 * parameters.chain_length, 3 * parameters.chain_length) = parameters.j1 * (self_energy_left(3, 3) - conj(self_energy_left(3, 3)));

        coupling_right(parameters.chain_length - 1, parameters.chain_length - 1) = parameters.j1 *
		(self_energy_right(0, 0) - conj(self_energy_right(0, 0)));
        
        coupling_right(2 * parameters.chain_length - 1, 2 * parameters.chain_length - 1) = parameters.j1  * 
            (self_energy_right(1, 1) - conj(self_energy_right(1, 1)));

        coupling_right(3 * parameters.chain_length  - 1, 3 * parameters.chain_length - 1) = parameters.j1 *
            (self_energy_right(2, 2) - conj(self_energy_right(2, 2)));

        coupling_right(4 * parameters.chain_length  - 1, 4 * parameters.chain_length - 1) = parameters.j1 *
            (self_energy_right(3, 3) - conj(self_energy_right(3, 3)));
}


void get_landauer_buttiker_current(const Parameters& parameters,
    const std::vector<dcomp>& transmission_up, const std::vector<dcomp>& transmission_down,
    double* current_up, double* current_down, const int votlage_step)
{
	
	for (int r = 0; r < parameters.steps_myid; r++) {
		*current_up -= (parameters.delta_energy * transmission_up.at(r)
		    * (fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)) + parameters.voltage_l[votlage_step],
		           parameters)
		        - fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid))
		                + parameters.voltage_r[votlage_step],
		            parameters))).real();
		*current_down -= (parameters.delta_energy * transmission_down.at(r)
		    * (fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)) + parameters.voltage_l[votlage_step],
		           parameters)
		        - fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid))
		                + parameters.voltage_r[votlage_step],
		            parameters))).real();	
	}
}

void get_meir_wingreen_current(
    const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, const int voltage_step, double *current_left,
	double *current_right, std::vector<dcomp>& transmission_up, const std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian)
{
    int n_x =  parameters.num_kx_points; //number of k points to take in x direction
    int n_y =  parameters.num_ky_points; //number of k points to take in y direction

	Eigen::MatrixXcd coupling_left = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
	Eigen::MatrixXcd coupling_right = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
	if (parameters.wbl_approx == true) {
		get_coupling(parameters, leads.at(0).at(0).self_energy_left.at(0), 
		leads.at(0).at(0).self_energy_right.at(0), coupling_left, coupling_right, 0);
	}    

	double num_k_points = n_x * n_y;
	for (int kx_i = 0; kx_i < n_x; kx_i++) {
		for (int ky_i = 0; ky_i < n_y; ky_i++) {
			Interacting_GF gf_interacting(parameters, self_energy_mb,
			    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
			    voltage_step, hamiltonian.at(kx_i).at(ky_i));

			std::vector<Eigen::MatrixXcd> gf_lesser(parameters.steps_myid,
			    Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));

			if (parameters.noneq_test == true) {
				get_gf_lesser_fd(parameters, gf_interacting.interacting_gf, gf_lesser);
			} else {
				get_gf_lesser_non_eq(parameters, gf_interacting.interacting_gf, self_energy_mb_lesser,
				    leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
				    gf_lesser, voltage_step);
			}

			dcomp current_k_resolved_left, current_k_resolved_right;

			get_k_dependent_transmission(parameters, gf_interacting.interacting_gf, coupling_left, coupling_right, transmission_up);

			get_meir_wingreen_k_dependent_current(parameters, gf_interacting.interacting_gf,
			    gf_lesser, leads.at(kx_i).at(ky_i).self_energy_left,
			    leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, &current_k_resolved_left, &current_k_resolved_right);

			*current_left -= (current_k_resolved_left).real() / num_k_points;
			*current_right -= (current_k_resolved_right).real() / num_k_points;
		}
	}
}





void get_meir_wingreen_k_dependent_current(const Parameters& parameters,
    std::vector<Eigen::MatrixXcd>& green_function,
    std::vector<Eigen::MatrixXcd>& green_function_lesser, const std::vector<Eigen::MatrixXcd>& self_energy_left,
    const std::vector<Eigen::MatrixXcd>& self_energy_right, const int voltage_step, dcomp* current_left, dcomp* current_right)
{

	dcomp trace_left = 0.0, trace_right = 0.0;

	//std::ostringstream oss;
	//oss << "textfiles/" << parameters.myid << ".integrand_file.txt";
	//std::string var = oss.str();
	//std::ofstream integrand_file;
	//integrand_file.open(var);



	Eigen::MatrixXcd coupling_left = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
	Eigen::MatrixXcd coupling_right = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);

	if (parameters.wbl_approx == true) { //we only need to do this once for the wide band limit
		get_coupling(parameters, self_energy_left.at(0), 
			self_energy_right.at(0), coupling_left, coupling_right, 0);
	}


	for (int r = 0; r < parameters.steps_myid; r++) {
		trace_left = 0, trace_right = 0;
		if (parameters.wbl_approx == false) {
			get_coupling(parameters, self_energy_left.at(r), 
				self_energy_right.at(r), coupling_left, coupling_right, r + parameters.start.at(parameters.myid));
		}
		//this formula is from PHYSICAL REVIEW B 72, 125114 2005
		for (int i = 0; i < 4 * parameters.chain_length; i++){

				if (parameters.noneq_test == true) {
					trace_left += fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)),
		        		parameters) * coupling_left(i, i) * parameters.j1 * (green_function.at(r)(i, i) - conj(green_function.at(r)(i, i)));

					trace_right += fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)),
		        		parameters) * coupling_right(i, i) * parameters.j1 * (green_function.at(r)(i, i) - conj(green_function.at(r)(i, i)));	
				} else {
					trace_left += fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)) - parameters.voltage_l.at(voltage_step),
		        		parameters) * coupling_left(i, i) * parameters.j1 * (green_function.at(r)(i, i) - conj(green_function.at(r)(i, i)));

					trace_right += fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)) - parameters.voltage_r.at(voltage_step),
		        		parameters) * coupling_right(i, i) * parameters.j1 * (green_function.at(r)(i, i) - conj(green_function.at(r)(i, i)));
				}

	
				trace_left += parameters.j1 * coupling_left(i, i) * green_function_lesser.at(r)(i, i);
				trace_right += parameters.j1 * coupling_right(i, i) * green_function_lesser.at(r)(i, i);

		}



		//integrand_file << parameters.energy.at(r) << "  "
		//		<< trace_left.real() << "  "
		//		<< trace_right.real() <<"\n";
				
		*current_left -= parameters.delta_energy * trace_left;
		*current_right -= parameters.delta_energy * trace_right;
	}

	//integrand_file.close();
}
