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
#include "aim.h"
#include "utilis.h"
#include "nca.h"


void get_difference(const Parameters &parameters, MatrixVectorType &gf_local_up, MatrixVectorType &old_green_function,
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

void dmft(const Parameters &parameters, const int voltage_step, 
        std::vector<std::vector<dcomp>> &self_energy_mb_up, std::vector<std::vector<dcomp>> &self_energy_mb_down, 
        std::vector<std::vector<dcomp>> &self_energy_mb_lesser_up, std::vector<std::vector<dcomp>> &self_energy_mb_lesser_down,
        std::vector<std::vector<dcomp>> &self_energy_mb_greater_up, std::vector<std::vector<dcomp>> &self_energy_mb_greater_down,
        MatrixVectorType &gf_local_up, MatrixVectorType &gf_local_down,
        MatrixVectorType &gf_local_lesser_up, MatrixVectorType &gf_local_lesser_down,
        MatrixVectorType &gf_local_greater_up, MatrixVectorType &gf_local_greater_down,
        const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<double> &spins_occup, const std::vector<MatrixVectorType> &hamiltonian_up,
		const std::vector<MatrixVectorType> &hamiltonian_down)
{
	double difference = std::numeric_limits<double>::infinity();
	int index, count = 0;

	std::vector<std::vector<dcomp>> old_self_energy_mb(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid, 100));

	if (parameters.spin_polarised == true) {
		std::vector<dcomp> diag_gf_local_up(parameters.steps_myid), diag_gf_local_down(parameters.steps_myid), diag_gf_local_lesser_up(parameters.steps_myid),
	    	diag_gf_local_lesser_down(parameters.steps_myid), diag_gf_local_greater_up,
	    	diag_gf_local_greater_down, impurity_self_energy_up(parameters.steps_myid), 
			impurity_self_energy_down(parameters.steps_myid), impurity_self_energy_lesser_up(parameters.steps_myid), 
			impurity_self_energy_lesser_down(parameters.steps_myid),  impurity_self_energy_greater_up, 
			impurity_self_energy_greater_down;

			if (parameters.impurity_solver == 3) {
				diag_gf_local_greater_up.resize(parameters.steps_myid);
	    		diag_gf_local_greater_down.resize(parameters.steps_myid);
				impurity_self_energy_greater_up.resize(parameters.steps_myid); 
				impurity_self_energy_greater_down.resize(parameters.steps_myid);
			}

		while (difference > parameters.convergence && count < parameters.self_consistent_steps) {
			//MPI_Barrier(MPI_COMM_WORLD);
			//std::cout << std::setprecision(15) << "The difference is " << difference << ". The count is " << count << std::endl;
			get_difference_self_energy(parameters, self_energy_mb_up, old_self_energy_mb, difference, index);

			if (parameters.myid == 0) std::cout << std::setprecision(15) << "The difference is " << difference << ". The count is " << count << std::endl;
			if (difference < parameters.convergence) break;

			for (int i = 0; i < 4 * parameters.chain_length; i++) {  //we only do the dmft loop over the correlated metal.
				if (parameters.atom_type.at(i) == 0) continue;//this allows us to only apply correlation on the metallic atoms or half metallic atoms. 
				//this are types 1 and 2 and hence continue will meaning all insulting layers are skipped.
				//this is only passing the part of the green function that each process is dealing with.
				for (int r = 0; r < parameters.steps_myid; r++) {//getting the diagonal part of the green function.
					diag_gf_local_up.at(r) = gf_local_up.at(r)(i, i);
					diag_gf_local_down.at(r) = gf_local_down.at(r)(i, i);
					diag_gf_local_lesser_up.at(r) = gf_local_lesser_up.at(r)(i, i);
					diag_gf_local_lesser_down.at(r) = gf_local_lesser_down.at(r)(i, i);
					impurity_self_energy_up.at(r) = self_energy_mb_up.at(i).at(r);
					impurity_self_energy_down.at(r) = self_energy_mb_down.at(i).at(r);
					impurity_self_energy_lesser_up.at(r) = self_energy_mb_lesser_up.at(i).at(r);
					impurity_self_energy_lesser_down.at(r) = self_energy_mb_lesser_down.at(i).at(r);
				}

				if (parameters.impurity_solver == 3) {//need to to the greater gf for nca
					for (int r = 0; r < parameters.steps_myid; r++) {
						diag_gf_local_greater_up.at(r) = gf_local_greater_up.at(r)(i, i);
						diag_gf_local_greater_down.at(r) = gf_local_greater_down.at(r)(i, i);
						impurity_self_energy_greater_up.at(r) = self_energy_mb_greater_up.at(i).at(r);
						impurity_self_energy_greater_down.at(r) = self_energy_mb_greater_down.at(i).at(r);
					}
				}

				//MPI_Allgather(&diag_gf_local_up_myid, parameters.steps_myid, MPI_DOUBLE_COMPLEX, &diag_gf_local_up, parameters.steps_myid, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
				if (parameters.myid == 0) std::cout << "atom which we put on correlation is " << i << std::endl;

    			AIM aim_up(parameters, diag_gf_local_up, diag_gf_local_lesser_up, diag_gf_local_greater_up,
					impurity_self_energy_up, impurity_self_energy_lesser_up,impurity_self_energy_greater_up, voltage_step);
    			AIM aim_down(parameters, diag_gf_local_down, diag_gf_local_lesser_down, diag_gf_local_greater_down,
				 	impurity_self_energy_down, impurity_self_energy_lesser_down, impurity_self_energy_greater_down, voltage_step);

				if (parameters.impurity_solver != 3) {
					impurity_solver_sigma_2(parameters, voltage_step, aim_up, aim_down, &spins_occup.at(i), &spins_occup.at(i + 4 * parameters.chain_length));
				} else {
					impurity_solver_nca(parameters, voltage_step, aim_up, aim_down);
				}
				
				if (parameters.myid == 0) {
					std::cout << "\n \n";
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

				if (parameters.impurity_solver == 3) {
					if(count == 0){
    	            	for (int r = 0; r < parameters.steps_myid; r++) {
    	            	    self_energy_mb_greater_up.at(i).at(r) = parameters.j1 * aim_up.self_energy_mb_greater.at(r);
    	            	    self_energy_mb_greater_down.at(i).at(r) = parameters.j1 * aim_down.self_energy_mb_greater.at(r);
    	            	}
    	        	} else {
    	            	for (int r = 0; r < parameters.steps_myid; r++) {
    	                	self_energy_mb_greater_up.at(i).at(r) = (parameters.j1 * aim_up.self_energy_mb_greater.at(r)) * 0.3 + self_energy_mb_greater_up.at(i).at(r) * 0.7;
    	                	self_energy_mb_greater_down.at(i).at(r) = (parameters.j1 * aim_down.self_energy_mb_greater.at(r)) * 0.3 + self_energy_mb_greater_down.at(i).at(r) * 0.7;
    	            	}
    	        	}
				}
			}
			
			if (parameters.impurity_solver == 3) {//this calculates g^> as well which is required for the nca.
				get_local_gf_r_greater_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, self_energy_mb_greater_up, leads, gf_local_up, gf_local_lesser_up,
				 	gf_local_greater_up, voltage_step, hamiltonian_up);
				get_local_gf_r_greater_lesser(parameters, self_energy_mb_down, self_energy_mb_lesser_down, self_energy_mb_greater_down, leads,
					 gf_local_down, gf_local_lesser_down, gf_local_greater_down, voltage_step, hamiltonian_down);	
			} else {//only need gf_retarded and gf_lesser for sigma2
				get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, voltage_step, hamiltonian_up);
				get_local_gf_r_and_lesser(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, gf_local_down, gf_local_lesser_down, voltage_step, hamiltonian_down);	
			}

			count++;
		}
	} else { //spin up is the same as spin down. only do the up channel.

		std::vector<dcomp> diag_gf_local_up(parameters.steps_myid), diag_gf_local_lesser_up(parameters.steps_myid),
	    	diag_gf_local_greater_up, impurity_self_energy_up(parameters.steps_myid), impurity_self_energy_lesser_up(parameters.steps_myid), 
			impurity_self_energy_greater_up;

			if (parameters.impurity_solver == 3) {
				diag_gf_local_greater_up.resize(parameters.steps_myid);
				impurity_self_energy_greater_up.resize(parameters.steps_myid); 
			}

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
				if (parameters.atom_type.at(i) == 0) continue; //this allows us to only apply correlation on the metallic atoms or half metallic atoms. 
				//this are types 1 and 2 and hence continue will meaning all insulting layers are skipped.
				//this is only passing the part of the green function that each process is dealing with.
				for (int r = 0; r < parameters.steps_myid; r++) {//getting the diagonal part of the GF
					diag_gf_local_up.at(r) = gf_local_up.at(r)(i, i);
					diag_gf_local_lesser_up.at(r) = gf_local_lesser_up.at(r)(i, i);
					impurity_self_energy_up.at(r) = self_energy_mb_up.at(i).at(r);
					impurity_self_energy_lesser_up.at(r) = self_energy_mb_lesser_up.at(i).at(r);
					//write_to_file(parameters, diag_gf_local_lesser_up, "diag_gf_local_lesser_up", 0);
				}


				if (parameters.impurity_solver == 3) {
					for (int r = 0; r < parameters.steps_myid; r++) {
						diag_gf_local_greater_up.at(r) = gf_local_greater_up.at(r)(i, i);
						impurity_self_energy_greater_up.at(r) = self_energy_mb_greater_up.at(i).at(r);
					}
					//write_to_file(parameters, diag_gf_local_greater_up, "diag_gf_local_greater_up", 0);
				}

				//MPI_Allgather(&diag_gf_local_up_myid, parameters.steps_myid, MPI_DOUBLE_COMPLEX, &diag_gf_local_up, parameters.steps_myid, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
				if (parameters.myid == 0) std::cout << "atom which we put on correlation is " << i << std::endl;

    			AIM aim_up(parameters, diag_gf_local_up, diag_gf_local_lesser_up, diag_gf_local_greater_up,
					impurity_self_energy_up, impurity_self_energy_lesser_up,impurity_self_energy_greater_up, voltage_step);

				if (parameters.impurity_solver != 3) {//will spend it to sigma2 to do brute force or kramer-kronig
					impurity_solver_sigma_2(parameters, voltage_step, aim_up, aim_up, &spins_occup.at(i), &spins_occup.at(i + 4 * parameters.chain_length));
				} else if (parameters.impurity_solver == 3) {// will do nca
					impurity_solver_nca(parameters, voltage_step, aim_up, aim_up);
				}
				
    	        if (count == 0){//this is so the difference in each self consistent is minimised. Helps convergence
    	            for (int r = 0; r < parameters.steps_myid; r++) {
    	                self_energy_mb_up.at(i).at(r) = aim_up.self_energy_mb_retarded.at(r);
    	                self_energy_mb_lesser_up.at(i).at(r) = parameters.j1 * aim_up.self_energy_mb_lesser.at(r);
    	            }
    	        } else {
    	            for (int r = 0; r < parameters.steps_myid; r++) {
    	                self_energy_mb_up.at(i).at(r) = aim_up.self_energy_mb_retarded.at(r) * 0.3 + (self_energy_mb_up.at(i).at(r)) * 0.7;
    	                self_energy_mb_lesser_up.at(i).at(r) = (parameters.j1 * aim_up.self_energy_mb_lesser.at(r)) * 0.3 + self_energy_mb_lesser_up.at(i).at(r) * 0.7;
    	            }
    	        }

				if (parameters.impurity_solver == 3) {// need to do this for g^> for the nca
					if (count == 0){//this is so the difference in each self consistent is minimised. Helps convergence
    	            	for (int r = 0; r < parameters.steps_myid; r++) {
    	            	    self_energy_mb_greater_up.at(i).at(r) = parameters.j1 * aim_up.self_energy_mb_greater.at(r);
    	            	}
    	        	} else {
    	            	for (int r = 0; r < parameters.steps_myid; r++) {
    	                	self_energy_mb_greater_up.at(i).at(r) = (parameters.j1 * aim_up.self_energy_mb_greater.at(r) + self_energy_mb_greater_up.at(i).at(r)) * 0.5;
    	            	}
    	        	}
				}
			}

			if (parameters.impurity_solver == 3) {//this calculates g^> as well which is required for the nca.
				get_local_gf_r_greater_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, self_energy_mb_greater_up, leads, gf_local_up, gf_local_lesser_up,
			 		gf_local_greater_up, voltage_step, hamiltonian_up);
			} else {//only need gf_retarded and gf_lesser for sigma2
				get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, voltage_step, hamiltonian_up);
			}
			count++;
		}
	}
}
