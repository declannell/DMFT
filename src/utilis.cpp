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
#include "AIM.h"
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

void write_to_file(Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_up, std::vector<Eigen::MatrixXcd> &gf_down, std::string filename, int voltage_step){
	for (int i = 0; i < parameters.chain_length * 2; i++){
		MPI_Barrier(MPI_COMM_WORLD);
		std::vector<dcomp> vec_1_up, vec_2_up;
		std::vector<dcomp> vec_1_down, vec_2_down;

		if (parameters.myid == 0) {
			//std::cout << "rank " << parameters.myid << " enters where parameters.myid == 0 \n";
			vec_1_up.resize(parameters.steps);
			vec_1_down.resize(parameters.steps);
			for (int r = 0; r < parameters.steps_myid; r ++){
				vec_1_up.at(r) = gf_up.at(r)(i, i);  
				vec_1_down.at(r) = gf_down.at(r)(i, i);
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
			ossgf << "textfiles/" << voltage_step << "." << i << "." << filename;
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

void write_to_file(Parameters &parameters, std::vector<dcomp> &gf_up, std::vector<dcomp> &gf_down, std::string filename, int voltage_step){
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
			ossgf << "textfiles/" << voltage_step << "." << filename;
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


void write_to_file(Parameters &parameters, std::vector<std::vector<dcomp>> &se_up, std::vector<std::vector<dcomp>> &se_down, std::string filename, int voltage_step){
	for (int i = 0; i < parameters.chain_length * 2; i++){

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
			ossgf << "textfiles/" << voltage_step << "." << i << "." << filename;
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

void distribute_to_procs(const Parameters &parameters, std::vector<double> &vec_1, std::vector<double> &vec_2){
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