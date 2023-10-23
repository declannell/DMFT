#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include "dmft.h"
#include "transport.h"
#include "analytic_gf.h"
#include "aim.h"
#include <mpi.h>
#include "utilis.h"

void decomp(int steps, int size, int myid, int *s, int *e) {
    int remainder = steps % size; 
    int steps_per_proc = steps / size; //rounds towards 0
	if (myid < remainder) {
		*s = myid * (steps_per_proc + 1);
		*e = *s + steps_per_proc;
	} else {
		*s = myid * (steps_per_proc) + remainder;
		*e = *s + steps_per_proc - 1;
	}
}

void get_momentum_vectors(std::vector<double> &kx, std::vector<double> &ky, Parameters &parameters) {
	for (int i = 0; i < parameters.num_kx_points; i++) {
		if (parameters.num_kx_points != 1) {
			kx.at(i) = 2 * M_PI * i / parameters.num_kx_points;
		} else if (parameters.num_kx_points == 1) {
			kx.at(i) = M_PI / 2.0;
		}
	}

	for (int i = 0; i < parameters.num_ky_points; i++) {
		if (parameters.num_ky_points != 1) {
			ky.at(i) = 2 * M_PI * i / parameters.num_ky_points;
		} else if (parameters.num_ky_points == 1) {
			ky.at(i) = M_PI / 2.0;
		}
	}
}

void write_to_file(const Parameters &parameters, std::vector<double> &gf_up, std::string filename, int voltage_step){
	MPI_Barrier(MPI_COMM_WORLD);
	std::vector<double> vec_1_up;
	std::vector<double> vec_2_up;

	if (parameters.myid == 0) {
		//std::cout << "rank " << parameters.myid << " enters where parameters.myid == 0 \n";
		vec_1_up.resize(parameters.steps);
		for (int r = 0; r < parameters.steps_myid; r ++){
			vec_1_up.at(r) = gf_up.at(r);  
		}
		for (int a = 1; a < parameters.size; a++){
			MPI_Recv(&vec_1_up.at(parameters.start.at(a)), parameters.steps_proc.at(a), MPI_DOUBLE, a, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			//std::cout << "I, rank 0, recieved part of vec_1 from rank " << a << std::endl; 
				//for (int r = parameters.start.at(a); r < parameters.start.at(a) + parameters.steps_proc.at(a); r ++){
			//	std::cout << "This part has a value of " << vec_1.at(r) << " " << r << std::endl; 
			//}
		}
	} else {
		//std::cout << "rank " << parameters.myid << " enters where parameters.myid != 0 \n";
		vec_2_up.resize(parameters.steps_myid);
		for (int r = 0; r < parameters.steps_myid; r++) {
			vec_2_up.at(r) = gf_up.at(r);
			//std::cout << "On rank " << parameters.myid << " vector has a value of " << vec_2.at(r)  << std::endl;
		}
		MPI_Send(&(vec_2_up.at(0)), parameters.steps_myid, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			//std::cout << "I, rank " << parameters.myid << " sent my part of the GF to rank 0 \n"; 
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (parameters.myid == 0) {
		//for (int r = 0; r < parameters.steps; r++) {
		//	std::cout << vec_1.at(r) << std::endl;
		//}
		std::ostringstream ossgf;
		ossgf << voltage_step <<  "." << filename;
		std::string var = ossgf.str();
		std::ofstream gf_local_file;
		gf_local_file.open(var);
		for (int r = 0; r < parameters.steps; r++) {
			gf_local_file << parameters.energy.at(r) << "  " << vec_1_up.at(r) << "  \n";
			              //<< gf_local_down.at(r)(i, i).real() << "   " << gf_local_down.at(r)(i, i).imag() << " \n"
			// std::cout << leads.self_energy_left.at(r) << "\n";
		}
		gf_local_file.close();					
	}
}

void write_to_file(const Parameters &parameters, std::vector<dcomp> &gf_up, std::string filename, int voltage_step){
	MPI_Barrier(MPI_COMM_WORLD);
	std::vector<dcomp> vec_1_up;
	std::vector<dcomp> vec_2_up;

	if (parameters.myid == 0) {
		//std::cout << "rank " << parameters.myid << " enters where parameters.myid == 0 \n";
		vec_1_up.resize(parameters.steps);
		for (int r = 0; r < parameters.steps_myid; r ++){
			vec_1_up.at(r) = gf_up.at(r);  
		}
		for (int a = 1; a < parameters.size; a++){
			MPI_Recv(&vec_1_up.at(parameters.start.at(a)), parameters.steps_proc.at(a), MPI_DOUBLE_COMPLEX, a, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			//std::cout << "I, rank 0, recieved part of vec_1 from rank " << a << std::endl; 
				//for (int r = parameters.start.at(a); r < parameters.start.at(a) + parameters.steps_proc.at(a); r ++){
			//	std::cout << "This part has a value of " << vec_1.at(r) << " " << r << std::endl; 
			//}
		}
	} else {
		//std::cout << "rank " << parameters.myid << " enters where parameters.myid != 0 \n";
		vec_2_up.resize(parameters.steps_myid);
		for (int r = 0; r < parameters.steps_myid; r++) {
			vec_2_up.at(r) = gf_up.at(r);
			//std::cout << "On rank " << parameters.myid << " vector has a value of " << vec_2.at(r)  << std::endl;
		}
		MPI_Send(&(vec_2_up.at(0)), parameters.steps_myid, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
			//std::cout << "I, rank " << parameters.myid << " sent my part of the GF to rank 0 \n"; 
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (parameters.myid == 0) {
		//for (int r = 0; r < parameters.steps; r++) {
		//	std::cout << vec_1.at(r) << std::endl;
		//}
		std::ostringstream ossgf;
		ossgf << voltage_step <<  "." << filename;
		std::string var = ossgf.str();
		std::ofstream gf_local_file;
		gf_local_file.open(var);
		for (int r = 0; r < parameters.steps; r++) {
			gf_local_file << parameters.energy.at(r) << "  " << vec_1_up.at(r).real() << " " << vec_1_up.at(r).imag() << "  \n";
		}
		gf_local_file.close();					
	}
}


void write_to_file(const Parameters &parameters, MatrixVectorType &gf_up, MatrixVectorType &gf_down, std::string filename, int voltage_step){
	for (int i = 0; i < parameters.chain_length * 4; i++){
		for (int j = 0; j < parameters.chain_length * 4; j++) {
			MPI_Barrier(MPI_COMM_WORLD);
			std::vector<dcomp> vec_1_up, vec_2_up;
			std::vector<dcomp> vec_1_down, vec_2_down;

			if (parameters.myid == 0) {
				//std::cout << "rank " << parameters.myid << " enters where parameters.myid == 0 \n";
				vec_1_up.resize(parameters.steps);
				vec_1_down.resize(parameters.steps);
				for (int r = 0; r < parameters.steps_myid; r ++){
					vec_1_up.at(r) = gf_up.at(r)(i, j);  
					vec_1_down.at(r) = gf_down.at(r)(i, j);
				}
				for (int a = 1; a < parameters.size; a++){
					MPI_Recv(&vec_1_up.at(parameters.start.at(a)), parameters.steps_proc.at(a), MPI_DOUBLE_COMPLEX, a, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(&vec_1_down.at(parameters.start.at(a)), parameters.steps_proc.at(a), MPI_DOUBLE_COMPLEX, a, i + 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					//std::cout << "I, rank 0, recieved part of vec_1 from rank " << a << std::endl; 
					//for (int r = parameters.start.at(a); r < parameters.start.at(a) + parameters.steps_proc.at(a); r ++){
					//	std::cout << "This part has a value of " << vec_1.at(r) << " " << r << std::endl; 
					//}
				}
			} else {
				//std::cout << "rank " << parameters.myid << " enters where parameters.myid != 0 \n";
				vec_2_up.resize(parameters.steps_myid);
				vec_2_down.resize(parameters.steps_myid);
				for (int r = 0; r < parameters.steps_myid; r++) {
					vec_2_up.at(r) = gf_up.at(r)(i, i);
					vec_2_down.at(r) = gf_down.at(r)(i, i);
					//std::cout << "On rank " << parameters.myid << " vector has a value of " << vec_2.at(r)  << std::endl;
				}
				MPI_Send(&(vec_2_up.at(0)), parameters.steps_myid, MPI_DOUBLE_COMPLEX, 0, i, MPI_COMM_WORLD);
				MPI_Send(&(vec_2_down.at(0)), parameters.steps_myid, MPI_DOUBLE_COMPLEX, 0, i + 100, MPI_COMM_WORLD);
				//std::cout << "I, rank " << parameters.myid << " sent my part of the GF to rank 0 \n"; 
			}

			MPI_Barrier(MPI_COMM_WORLD);
			if (parameters.myid == 0) {
				//for (int r = 0; r < parameters.steps; r++) {
				//	std::cout << vec_1.at(r) << std::endl;
				//}
				std::ostringstream ossgf;
				ossgf << voltage_step << "." << i << "_" << j << "." << filename;
				std::string var = ossgf.str();
				std::ofstream gf_local_file;
				gf_local_file.open(var);
				for (int r = 0; r < parameters.steps; r++) {
					gf_local_file << parameters.energy.at(r) << "  " << vec_1_up.at(r).real() << "   " << vec_1_up.at(r).imag() << " "
						<< vec_1_down.at(r).real() << "   " << vec_1_down.at(r).imag() << "  \n";
					              //<< gf_local_down.at(r)(i, i).real() << "   " << gf_local_down.at(r)(i, i).imag() << " \n"
					// std::cout << leads.self_energy_left.at(r) << "\n";
				}
				gf_local_file.close();					
			}
		}
	}
}

void write_to_file(const Parameters &parameters, std::vector<dcomp> &gf_up, std::vector<dcomp> &gf_down, std::string filename, int voltage_step){
		MPI_Barrier(MPI_COMM_WORLD);
		std::vector<dcomp> vec_1_up, vec_2_up;
		std::vector<dcomp> vec_1_down, vec_2_down;

		if (parameters.myid == 0) {
			//std::cout << "rank " << parameters.myid << " enters where parameters.myid == 0 \n";
			vec_1_up.resize(parameters.steps);
			vec_1_down.resize(parameters.steps);
			for (int r = 0; r < parameters.steps_myid; r ++){
				vec_1_up.at(r) = gf_up.at(r);  
				vec_1_down.at(r) = gf_down.at(r);
			}
			for (int a = 1; a < parameters.size; a++){
				MPI_Recv(&vec_1_up.at(parameters.start.at(a)), parameters.steps_proc.at(a), MPI_DOUBLE_COMPLEX, a, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&vec_1_down.at(parameters.start.at(a)), parameters.steps_proc.at(a), MPI_DOUBLE_COMPLEX, a, 300 + 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//std::cout << "I, rank 0, recieved part of vec_1 from rank " << a << std::endl; 
				//for (int r = parameters.start.at(a); r < parameters.start.at(a) + parameters.steps_proc.at(a); r ++){
				//	std::cout << "This part has a value of " << vec_1.at(r) << " " << r << std::endl; 
				//}
			}
		} else {
			//std::cout << "rank " << parameters.myid << " enters where parameters.myid != 0 \n";
			vec_2_up.resize(parameters.steps_myid);
			vec_2_down.resize(parameters.steps_myid);
			for (int r = 0; r < parameters.steps_myid; r++) {
				vec_2_up.at(r) = gf_up.at(r);
				vec_2_down.at(r) = gf_down.at(r);
				//std::cout << "On rank " << parameters.myid << " vector has a value of " << vec_2.at(r)  << std::endl;
			}
			MPI_Send(&(vec_2_up.at(0)), parameters.steps_myid, MPI_DOUBLE_COMPLEX, 0, 300, MPI_COMM_WORLD);
			MPI_Send(&(vec_2_down.at(0)), parameters.steps_myid, MPI_DOUBLE_COMPLEX, 0, 300 + 100, MPI_COMM_WORLD);
			//std::cout << "I, rank " << parameters.myid << " sent my part of the GF to rank 0 \n"; 
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (parameters.myid == 0) {
			//for (int r = 0; r < parameters.steps; r++) {
			//	std::cout << vec_1.at(r) << std::endl;
			//}
			std::ostringstream ossgf;
			ossgf << voltage_step << "." << filename;
			std::string var = ossgf.str();
			std::ofstream gf_local_file;
			gf_local_file.open(var);
			for (int r = 0; r < parameters.steps; r++) {
				gf_local_file << parameters.energy.at(r) << "  " << vec_1_up.at(r).real() <<  " "
					<< vec_1_down.at(r).real() << "  \n";
				              //<< gf_local_down.at(r)(i, i).real() << "   " << gf_local_down.at(r)(i, i).imag() << " \n"
				// std::cout << leads.self_energy_left.at(r) << "\n";
			}
			gf_local_file.close();					
		}
}


void write_to_file(const Parameters &parameters, std::vector<std::vector<dcomp>> &se_up, std::vector<std::vector<dcomp>> &se_down, std::string filename, int voltage_step){
	for (int i = 0; i < parameters.chain_length * 4; i++){

		if (parameters.atom_type.at(i) == 0){
			continue;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		std::vector<dcomp> vec_1_up, vec_2_up;
		std::vector<dcomp> vec_1_down, vec_2_down;

		if (parameters.myid == 0) {
			//std::cout << "rank " << parameters.myid << " enters where parameters.myid == 0 \n";
			vec_1_up.resize(parameters.steps);
			vec_1_down.resize(parameters.steps);
			for (int r = 0; r < parameters.steps_myid; r ++){
				vec_1_up.at(r) = se_up.at(i).at(r);  
				vec_1_down.at(r) = se_down.at(i).at(r);
			}
			for (int a = 1; a < parameters.size; a++){
				MPI_Recv(&vec_1_up.at(parameters.start.at(a)), parameters.steps_proc.at(a), MPI_DOUBLE_COMPLEX, a, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&vec_1_down.at(parameters.start.at(a)), parameters.steps_proc.at(a), MPI_DOUBLE_COMPLEX, a, i + 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//std::cout << "I, rank 0, recieved part of vec_1 from rank " << a << std::endl; 
				//for (int r = parameters.start.at(a); r < parameters.start.at(a) + parameters.steps_proc.at(a); r ++){
				//	std::cout << "This part has a value of " << vec_1.at(r) << " " << r << std::endl; 
				//}
			}
		} else {
			//std::cout << "rank " << parameters.myid << " enters where parameters.myid != 0 \n";
			vec_2_up.resize(parameters.steps_myid);
			vec_2_down.resize(parameters.steps_myid);
			for (int r = 0; r < parameters.steps_myid; r++) {
				vec_2_up.at(r) = se_up.at(i).at(r);
				vec_2_down.at(r) = se_down.at(i).at(r);
				//std::cout << "On rank " << parameters.myid << " vector has a value of " << vec_2.at(r)  << std::endl;
			}
			MPI_Send(&(vec_2_up.at(0)), parameters.steps_myid, MPI_DOUBLE_COMPLEX, 0, i, MPI_COMM_WORLD);
			MPI_Send(&(vec_2_down.at(0)), parameters.steps_myid, MPI_DOUBLE_COMPLEX, 0, i + 100, MPI_COMM_WORLD);
			//std::cout << "I, rank " << parameters.myid << " sent my part of the GF to rank 0 \n"; 
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (parameters.myid == 0) {
			//for (int r = 0; r < parameters.steps; r++) {
			//	std::cout << vec_1.at(r) << std::endl;
			//}
			std::ostringstream ossgf;
			ossgf << voltage_step << "." << i << "." << filename;
			std::string var = ossgf.str();
			std::ofstream gf_local_file;
			gf_local_file.open(var);
			for (int r = 0; r < parameters.steps; r++) {
				gf_local_file << parameters.energy.at(r) << "  " << vec_1_up.at(r).real() << "   " << vec_1_up.at(r).imag() << " "
					<< vec_1_down.at(r).real() << "   " << vec_1_down.at(r).imag() << "  \n";
				              //<< gf_local_down.at(r)(i, i).real() << "   " << gf_local_down.at(r)(i, i).imag() << " \n"
				// std::cout << leads.self_energy_left.at(r) << "\n";
			}
			gf_local_file.close();					
		}
	}
}

void distribute_to_procs(const Parameters &parameters, std::vector<dcomp> &vec_1, std::vector<dcomp> &vec_2){
		MPI_Barrier(MPI_COMM_WORLD);

		if (parameters.myid == 0) {
			//std::cout << "rank " << parameters.myid << " enters where parameters.myid == 0 \n";
			for (int r = 0; r < parameters.steps_myid; r ++){
				vec_1.at(r) = vec_2.at(r);  
			}
			for (int a = 1; a < parameters.size; a++){
				MPI_Recv(&vec_1.at(parameters.start.at(a)), parameters.steps_proc.at(a), MPI_DOUBLE_COMPLEX, a, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//std::cout << "I, rank 0, recieved part of vec_1 from rank " << a << std::endl; 
				//for (int r = parameters.start.at(a); r < parameters.start.at(a) + parameters.steps_proc.at(a); r ++){
				//	std::cout << "This part has a value of " << vec_1.at(r) << " " << r << std::endl; 
				//}
			}
		} else {
			//std::cout << "rank " << parameters.myid << " enters where parameters.myid != 0 \n";
			MPI_Send(&(vec_2.at(0)), parameters.steps_myid, MPI_DOUBLE_COMPLEX, 0, 200, MPI_COMM_WORLD);
			//std::cout << "I, rank " << parameters.myid << " sent my part of the GF to rank 0 \n"; 
		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&(vec_1.at(0)), parameters.steps, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

}

void distribute_to_procs(const Parameters &parameters, std::vector<double> &vec_1, const std::vector<double> &vec_2){
		MPI_Barrier(MPI_COMM_WORLD);
		if (parameters.myid == 0) {
			//std::cout << "rank " << parameters.myid << " enters where parameters.myid == 0 \n";
			for (int r = 0; r < parameters.steps_myid; r ++){
				vec_1.at(r) = vec_2.at(r);  
			}
			for (int a = 1; a < parameters.size; a++){
				MPI_Recv(&vec_1.at(parameters.start.at(a)), parameters.steps_proc.at(a), MPI_DOUBLE, a, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//for (int r = parameters.start.at(a); r < parameters.start.at(a) + parameters.steps_proc.at(a); r ++){
				//	std::cout << "This part has a value of " << vec_1.at(r) << " " << r << std::endl; 
				//}
			}
		} else {
			//std::cout << "rank " << parameters.myid << " enters where parameters.myid != 0 \n";
			MPI_Send(&(vec_2.at(0)), parameters.steps_myid, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&(vec_1.at(0)), parameters.steps, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


double absolute_value(double num1) {
	return std::sqrt((num1 ) * (num1));

}

double kramer_kronig_relation(const Parameters &parameters, std::vector<double> &impurity_self_energy_imag, int r) {
    double se_real = 0;
    for (int i = 0; i < parameters.steps; i++) {
        if (i != r) {
            se_real += impurity_self_energy_imag.at(i) / (parameters.energy.at(i) - parameters.energy.at(r));
        }
    }
	return se_real * parameters.delta_energy / M_PI;
}



void integrate_spectral(Parameters &parameters, MatrixVectorType &gf_local){
	for(int i = 0; i < 4 * parameters.chain_length; i++){
		double result = 0.0, result_reduced = 0.0;
		for (int r = 0; r < parameters.steps_myid; r++) {
			double spectral = (parameters.j1 * (gf_local.at(r)(i, i) - std::conj(gf_local.at(r)(i, i)))).real();
		
			result += spectral;
 		}
		result = result * parameters.delta_energy / (2.0 * M_PI);
		MPI_Reduce(&result, &result_reduced, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (parameters.myid == 0) {
			std::cout << "For the atom number "<< i << " the spectral function integrates to " << result_reduced << std::endl;
		}
	}
}

void get_occupation(Parameters  &parameters, MatrixVectorType & gf_local_lesser_up, 
	MatrixVectorType & gf_local_lesser_down, std::vector<double> &spins_occup) {

	for(int i = 0; i < 4 * parameters.chain_length; i++){
		double result_up = 0.0, result_reduced_up = 0.0,
			result_down = 0.0, result_reduced_down = 0.0;
		if (parameters.spin_polarised == true) {
			for (int r = 0; r < parameters.steps_myid; r++) {
				result_up += gf_local_lesser_up.at(r)(i, i).imag();
				result_down += gf_local_lesser_down.at(r)(i, i).imag();
			}
			result_up = result_up * parameters.delta_energy / (2.0 * M_PI);
			result_down = result_down * parameters.delta_energy / (2.0 * M_PI);
			MPI_Reduce(&result_up, &result_reduced_up, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&result_down, &result_reduced_down, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 		} else  {//spin_down = spin_up
			for (int r = 0; r < parameters.steps_myid; r++) {
				result_up += gf_local_lesser_up.at(r)(i, i).imag();
			}
			result_up = result_up * parameters.delta_energy / (2.0 * M_PI);
			MPI_Reduce(&result_up, &result_reduced_up, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			result_reduced_down = result_reduced_up;
 		}


		if (parameters.myid == 0) {
			spins_occup.at(i) = result_reduced_up;
			spins_occup.at(i + 4 *parameters.chain_length) = result_reduced_down;
			std::cout << "For the atom number "<< i << " the spin up occupation is " << result_reduced_up << ". The spin down occupation is "
				<< result_reduced_down << std::endl;
		}
	}
}


void get_dos(Parameters &parameters, std::vector<dcomp> &dos_up, std::vector<dcomp> &dos_down, std::vector<dcomp> &dos_up_ins, std::vector<dcomp> &dos_down_ins,
 	std::vector<dcomp> &dos_up_metal, std::vector<dcomp> &dos_down_metal, MatrixVectorType &gf_local_up, MatrixVectorType &gf_local_down) {
		for (int r = 0; r < parameters.steps_myid; r++) {
			for (int i = 0; i < 4 * parameters.chain_length; i++) {
				dos_up.at(r) += -gf_local_up.at(r)(i, i).imag();
				dos_down.at(r) += -gf_local_down.at(r)(i, i).imag();

				if (parameters.atom_type.at(i) == 0){
					dos_up_ins.at(r) += -gf_local_up.at(r)(i, i).imag();
					dos_down_ins.at(r) += -gf_local_down.at(r)(i, i).imag();
				} else if (parameters.atom_type.at(i) == 1) {
					dos_up_metal.at(r) += -gf_local_up.at(r)(i, i).imag();
					dos_down_metal.at(r) += -gf_local_down.at(r)(i, i).imag();					
				}
			}
		}
}
