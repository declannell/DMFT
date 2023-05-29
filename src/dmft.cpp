#include "dmft.h"
#include <mpi.h>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <vector>
#include "sigma_2.h"
#include <cstdlib>
#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"
#include "AIM.h"
#include "utilis.h"

void get_spin_occupation(const Parameters &parameters, const std::vector<double> &gf_lesser_up,
                        const std::vector<double> &gf_lesser_down, double *spin_up, double *spin_down)
{
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)(parameters.steps);
	double result_up = 0.0, result_down = 0.0;

	if (parameters.spin_polarised == true) {
		for (int r = 0; r < parameters.steps_myid; r++) {
		//std::cout << parameters.energy.at(r) << " " << gf_lesser_down.at(r).imag() <<  std::endl;
			if (r + parameters.start.at(parameters.myid) == 0 || r + parameters.start.at(parameters.myid) == parameters.steps - 1) {
			//std::cout << "I am rank " << parameters.myid << std::endl;
				result_up += (delta_energy / 2.0) * gf_lesser_up.at(r);
				result_down += (delta_energy / 2.0) * gf_lesser_down.at(r);
			} else {
				result_up += (delta_energy) * gf_lesser_up.at(r);
				result_down += (delta_energy) * gf_lesser_down.at(r);
			}
		}
		result_up = 1.0 / (2.0 * M_PI) * result_up;
		result_down = 1.0 / (2.0 * M_PI) * result_down;
		MPI_Allreduce(&result_up, spin_up, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&result_down, spin_down, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
	} else {
		for (int r = 0; r < parameters.steps_myid; r++) {
			if (r + parameters.start.at(parameters.myid) == 0 || r + parameters.start.at(parameters.myid) == parameters.steps - 1) {
				result_up += (delta_energy / 2.0) * gf_lesser_up.at(r);
			} else {
				result_up += (delta_energy) * gf_lesser_up.at(r);
			}
		}
		result_up = 1.0 / (2.0 * M_PI) * result_up;
		MPI_Allreduce(&result_up, spin_up, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		*spin_down = *spin_up;
	}
}

void get_difference(const Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &old_green_function,
                double &difference, int &index){
	double difference_proc = - std::numeric_limits<double>::infinity();
	double old_difference = 0;
	double real_difference = 0, imag_difference = 0;
	for (int r = 0; r < parameters.steps_myid; r++) {
		for (int i = 0; i < 4 * parameters.chain_length; i++) {
			for (int j = 0; j < 4 * parameters.chain_length; j++) {
				real_difference = absolute_value(gf_local_up.at(r)(i, j).real() - old_green_function.at(r)(i, j).real());
				imag_difference = absolute_value(gf_local_up.at(r)(i, j).imag() - old_green_function.at(r)(i, j).imag());
				//std::cout << gf_local_up.at(r)(i, j).real() << " " << old_green_function.at(r)(i, j).real() << std::endl;
				//std::cout << real_difference << "  " << imag_difference << "  "  << difference << "\n";
				difference_proc = std::max(difference_proc, std::max(real_difference, imag_difference));
				old_green_function.at(r)(i, j) = gf_local_up.at(r)(i, j);
				if (difference_proc > old_difference) {
					index = r;
				}
				old_difference = difference_proc;
			}
		}
		//std::cout <<"\n";
	}
	//std::cout << "I am rank " << parameters.myid << ". The difference for me is " << difference_proc << std::endl;
	//MPI_Allreduce would do the same thing.
	MPI_Reduce(&difference_proc, &difference, 1, MPI_DOUBLE, MPI_MAX , 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&difference, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void get_difference_self_energy(const Parameters &parameters, std::vector<std::vector<dcomp>> &self_energy_mb_up,
	 std::vector<std::vector<dcomp>> &old_self_energy_mb_up, double &difference, int &index){
	double difference_proc = - std::numeric_limits<double>::infinity();
	double old_difference = 0;
	double real_difference = 0, imag_difference = 0;
	for (int r = 0; r < parameters.steps_myid; r++) {
		for (int i = 0; i < 4 * parameters.chain_length; i++) {
			real_difference = absolute_value(self_energy_mb_up.at(i).at(r).real() - old_self_energy_mb_up.at(i).at(r).real());
			imag_difference = absolute_value(self_energy_mb_up.at(i).at(r).imag() - old_self_energy_mb_up.at(i).at(r).imag());
			//std::cout << gf_local_up.at(r)(i, j).real() << " " << old_green_function.at(r)(i, j).real() << std::endl;
			//std::cout << real_difference << "  " << imag_difference << "  "  << difference << "\n";
			difference_proc = std::max(difference_proc, std::max(real_difference, imag_difference));
			old_self_energy_mb_up.at(i).at(r) = self_energy_mb_up.at(i).at(r);
			if (difference_proc > old_difference) {
				index = r;
			}
			old_difference = difference_proc;
		}
		//std::cout <<"\n";
	}
	//std::cout << "I am rank " << parameters.myid << ". The difference for me is " << difference_proc << std::endl;
	//MPI_Allreduce would do the same thing.
	MPI_Reduce(&difference_proc, &difference, 1, MPI_DOUBLE, MPI_MAX , 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&difference, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


void impurity_solver(const Parameters &parameters, const int voltage_step, 
    AIM &aim_up, AIM &aim_down, double *spin_up, double *spin_down)
{
	get_spin_occupation(parameters, aim_up.dynamical_field_lesser, aim_down.dynamical_field_lesser, spin_up, spin_down);
	if (parameters.myid == 0) {
		std::cout << std::setprecision(15) << "The spin up occupancy is " << *spin_up << "\n";
		std::cout << "The spin down occupancy is " << *spin_down << "\n";
	}

	if (parameters.impurity_solver == 2) {// kramer kronig relation.
		std::cout << "using the kramer-kronig relation for second order perturbation theory\n";
		if (parameters.spin_polarised == true) {
			self_energy_2nd_order_kramers_kronig(parameters, aim_up, aim_down, voltage_step);
			self_energy_2nd_order_kramers_kronig(parameters, aim_down, aim_up, voltage_step);
		} else { //only need to do this for spin up. Note aim_down = aim_up
			self_energy_2nd_order_kramers_kronig(parameters, aim_up, aim_down, voltage_step);
		}
	}

	if (parameters.impurity_solver == 1) {//brute force sigma_2
		if (parameters.spin_polarised == true) {
			self_energy_2nd_order(parameters, aim_up, aim_down);
			self_energy_2nd_order(parameters, aim_down, aim_up);
		} else { //only need to do this for spin up. Note aim_down = aim_up
			self_energy_2nd_order(parameters, aim_up, aim_down);
		}
	}

	if (parameters.impurity_solver == 1 || 2 || 0) {//if mean field or either implementation of sigma_2 then add the first order term.
		if (parameters.spin_polarised == true) {
			for (int r = 0; r < parameters.steps_myid; r++) {
				aim_up.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * (*spin_down);
				aim_down.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * (*spin_up);
			} 
		} else {
			for (int r = 0; r < parameters.steps_myid; r++) {
				aim_up.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * (*spin_down);
			}	
		}
	}

	if (parameters.impurity_solver == 3) {
		
	}
}

void dmft(const Parameters &parameters, const int voltage_step, 
        std::vector<std::vector<dcomp>> &self_energy_mb_up, std::vector<std::vector<dcomp>> &self_energy_mb_down, 
        std::vector<std::vector<dcomp>> &self_energy_mb_lesser_up, std::vector<std::vector<dcomp>> &self_energy_mb_lesser_down,
        std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &gf_local_down,
        std::vector<Eigen::MatrixXcd> &gf_local_lesser_up, std::vector<Eigen::MatrixXcd> &gf_local_lesser_down,
        const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<double> &spins_occup, const std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian)
{
	double difference = std::numeric_limits<double>::infinity();
	int index, count = 0;

	std::vector<Eigen::MatrixXcd> old_green_function(parameters.steps_myid, Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
	std::vector<std::vector<dcomp>> old_self_energy_mb(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid, 100));

	if (parameters.spin_polarised == true) {
		std::vector<dcomp> diag_gf_local_up(parameters.steps_myid), diag_gf_local_down(parameters.steps_myid), diag_gf_local_lesser_up(parameters.steps_myid),
	    	diag_gf_local_lesser_down(parameters.steps_myid), impurity_self_energy_up(parameters.steps_myid), impurity_self_energy_down(parameters.steps_myid),
	    	impurity_self_energy_lesser_up(parameters.steps_myid), impurity_self_energy_lesser_down(parameters.steps_myid);

		while (difference > parameters.convergence && count < parameters.self_consistent_steps) {
			//MPI_Barrier(MPI_COMM_WORLD);
			//std::cout << std::setprecision(15) << "The difference is " << difference << ". The count is " << count << std::endl;
			get_difference_self_energy(parameters, self_energy_mb_up, old_self_energy_mb, difference, index);
			MPI_Barrier(MPI_COMM_WORLD);
			if (parameters.myid == 0) {
				std::cout << std::setprecision(15) << "The difference is " << difference << ". The count is " << count << std::endl;
			}

			if (difference < parameters.convergence) {
				break;
			}

			for (int i = 0; i < 4 * parameters.chain_length; i++) {  //we only do the dmft loop over the correlated metal.
				if (parameters.atom_type.at(i) == 0){
					continue;
				}
				//this is only passing the part of the green function that each process is dealing with.
				for (int r = 0; r < parameters.steps_myid; r++) {
					diag_gf_local_up.at(r) = gf_local_up.at(r)(i, i);
					diag_gf_local_down.at(r) = gf_local_down.at(r)(i, i);
					diag_gf_local_lesser_up.at(r) = gf_local_lesser_up.at(r)(i, i);
					diag_gf_local_lesser_down.at(r) = gf_local_lesser_down.at(r)(i, i);
					impurity_self_energy_up.at(r) = self_energy_mb_up.at(i).at(r);
					impurity_self_energy_down.at(r) = self_energy_mb_down.at(i).at(r);
					impurity_self_energy_lesser_up.at(r) = self_energy_mb_lesser_up.at(i).at(r);
					impurity_self_energy_lesser_down.at(r) = self_energy_mb_lesser_down.at(i).at(r);
				}

				//MPI_Allgather(&diag_gf_local_up_myid, parameters.steps_myid, MPI_DOUBLE_COMPLEX, &diag_gf_local_up, parameters.steps_myid, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
				if (parameters.myid == 0) {
					std::cout << "atom which we put on correlation is " << i << std::endl;
				}


    			AIM aim_up(parameters, diag_gf_local_up, diag_gf_local_lesser_up, impurity_self_energy_up, impurity_self_energy_lesser_up, voltage_step);
    			AIM aim_down(parameters, diag_gf_local_down, diag_gf_local_lesser_down, impurity_self_energy_down, impurity_self_energy_lesser_down, voltage_step);

				impurity_solver(parameters, voltage_step, aim_up, aim_down, &spins_occup.at(i), &spins_occup.at(i + 4 * parameters.chain_length));


				if (parameters.myid == 0) {
					std::cout << "AIM was created for atom " << i << std::endl;
				}

    	        if(count == 0){
    	            for (int r = 0; r < parameters.steps_myid; r++) {
    	                self_energy_mb_up.at(i).at(r) = aim_up.self_energy_mb_retarded.at(r);
						//std::cout << self_energy_mb_up.at(i).at(r) << " " << aim_up.self_energy_mb_retarded.at(r) << "\n";
    	                self_energy_mb_down.at(i).at(r) = aim_down.self_energy_mb_retarded.at(r);
    	                self_energy_mb_lesser_up.at(i).at(r) = parameters.j1 * aim_up.self_energy_mb_lesser.at(r);
    	                self_energy_mb_lesser_down.at(i).at(r) = parameters.j1 * aim_down.self_energy_mb_lesser.at(r);
    	            }
    	        } else {
    	            for (int r = 0; r < parameters.steps_myid; r++) {
    	                self_energy_mb_up.at(i).at(r) = (aim_up.self_energy_mb_retarded.at(r) + self_energy_mb_up.at(i).at(r)) * 0.5;
    	                self_energy_mb_down.at(i).at(r) = (aim_down.self_energy_mb_retarded.at(r) + self_energy_mb_down.at(i).at(r)) * 0.5;
    	                self_energy_mb_lesser_up.at(i).at(r) = (parameters.j1 * aim_up.self_energy_mb_lesser.at(r) + self_energy_mb_lesser_up.at(i).at(r)) * 0.5;
    	                self_energy_mb_lesser_down.at(i).at(r) = (parameters.j1 * aim_down.self_energy_mb_lesser.at(r) + self_energy_mb_lesser_down.at(i).at(r)) * 0.5;
    	            }
    	        }
			}

			if (parameters.noneq_test == true) {
				get_noneq_test(parameters, self_energy_mb_up, leads, gf_local_up, gf_local_lesser_up, voltage_step, hamiltonian);
				get_noneq_test(parameters, self_energy_mb_down, leads, gf_local_down, gf_local_lesser_down, voltage_step, hamiltonian);				
			} else {
				get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, voltage_step, hamiltonian);
				get_local_gf_r_and_lesser(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, gf_local_down, gf_local_lesser_down, voltage_step, hamiltonian);			
			}


			count++;
		}
	} else { //spin up is the same as spin down
		std::vector<dcomp> diag_gf_local_up(parameters.steps_myid), diag_gf_local_lesser_up(parameters.steps_myid),
			impurity_self_energy_up(parameters.steps_myid), impurity_self_energy_lesser_up(parameters.steps_myid);

		while (difference > parameters.convergence && count < parameters.self_consistent_steps) {
			//MPI_Barrier(MPI_COMM_WORLD);
			//std::cout << std::setprecision(15) << "The difference is " << difference << ". The count is " << count << std::endl;
			get_difference_self_energy(parameters, self_energy_mb_up, old_self_energy_mb, difference, index);
			MPI_Barrier(MPI_COMM_WORLD);
			if (parameters.myid == 0) {
				std::cout << std::setprecision(15) << "The difference is " << difference << ". The count is " << count << std::endl;
			}

			if (difference < parameters.convergence) {
				break;
			}

			for (int i = 0; i < 4 * parameters.chain_length; i++) {  //we only do the dmft loop over the correlated metal.
				if (parameters.atom_type.at(i) == 0){
					continue;
				}
				//this is only passing the part of the green function that each process is dealing with.
				for (int r = 0; r < parameters.steps_myid; r++) {
					diag_gf_local_up.at(r) = gf_local_up.at(r)(i, i);
					diag_gf_local_lesser_up.at(r) = gf_local_lesser_up.at(r)(i, i);
					impurity_self_energy_up.at(r) = self_energy_mb_up.at(i).at(r);
					impurity_self_energy_lesser_up.at(r) = self_energy_mb_lesser_up.at(i).at(r);
				}

				//MPI_Allgather(&diag_gf_local_up_myid, parameters.steps_myid, MPI_DOUBLE_COMPLEX, &diag_gf_local_up, parameters.steps_myid, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
				if (parameters.myid == 0) {
					std::cout << "atom which we put on correlation is " << i << std::endl;
				}


    			AIM aim_up(parameters, diag_gf_local_up, diag_gf_local_lesser_up, impurity_self_energy_up, impurity_self_energy_lesser_up, voltage_step);

				impurity_solver(parameters, voltage_step, aim_up, aim_up, &spins_occup.at(i), &spins_occup.at(i + 4 * parameters.chain_length));

				if (parameters.myid == 0) {
					std::cout << "AIM was created for atom " << i << std::endl;
				}

    	        if(count == 0){
    	            for (int r = 0; r < parameters.steps_myid; r++) {
    	                self_energy_mb_up.at(i).at(r) = aim_up.self_energy_mb_retarded.at(r);
    	                self_energy_mb_lesser_up.at(i).at(r) = parameters.j1 * aim_up.self_energy_mb_lesser.at(r);
    	            }
    	        } else {
    	            for (int r = 0; r < parameters.steps_myid; r++) {
    	                self_energy_mb_up.at(i).at(r) = (aim_up.self_energy_mb_retarded.at(r) + self_energy_mb_up.at(i).at(r)) * 0.5;
    	                self_energy_mb_lesser_up.at(i).at(r) = (parameters.j1 * aim_up.self_energy_mb_lesser.at(r) + self_energy_mb_lesser_up.at(i).at(r)) * 0.5;
    	            }
    	        }
			}

			if (parameters.noneq_test == true) {
				get_noneq_test(parameters, self_energy_mb_up, leads, gf_local_up, gf_local_lesser_up, voltage_step, hamiltonian);
			} else {
				get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, voltage_step, hamiltonian);
			}
			count++;
		}
	}
	
}
