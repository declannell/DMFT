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

void set_initial_spin(Parameters &parameters, std::vector<std::vector<dcomp>> &self_energy_mb_up, std::vector<std::vector<dcomp>> &self_energy_mb_down){
	for (int i = 0; i < 4 * parameters.chain_length; i++) {
		if (parameters.atom_type.at(i) == 1) {
			for (int r = 0; r < parameters.steps_myid; r++) {
				self_energy_mb_up.at(i).at(r) = parameters.spin_down_occup * parameters.hubbard_interaction;
				self_energy_mb_down.at(i).at(r) = parameters.spin_up_occup * parameters.hubbard_interaction;
			}
		}
	}
}

void integrate_spectral(Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local){
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
	for(int i = 0; i < 4 * parameters.chain_length; i++){
		double result = 0.0, result_reduced = 0.0;
		for (int r = 0; r < parameters.steps_myid; r++) {
			double spectral = (parameters.j1 * (gf_local.at(r)(i, i) - std::conj(gf_local.at(r)(i, i)))).real();
		
			result += spectral;
 		}
		result = result * delta_energy / (2.0 * M_PI);
		MPI_Reduce(&result, &result_reduced, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (parameters.myid == 0) {
			std::cout << "For the atom number "<< i << " the spectral function integrates to " << result_reduced << std::endl;
		}
	}
}

void get_occupation(Parameters  &parameters, std::vector<Eigen::MatrixXcd> & gf_local_lesser_up, 
	std::vector<Eigen::MatrixXcd> & gf_local_lesser_down, std::vector<double> &spins_occup) {
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;

	for(int i = 0; i < 4 * parameters.chain_length; i++){
		double result_up = 0.0, result_reduced_up = 0.0,
			result_down = 0.0, result_reduced_down = 0.0;
		if (parameters.spin_polarised == true) {
			for (int r = 0; r < parameters.steps_myid; r++) {
				result_up += gf_local_lesser_up.at(r)(i, i).imag();
				result_down += gf_local_lesser_down.at(r)(i, i).imag();
			}
			result_up = result_up * delta_energy / (2.0 * M_PI);
			result_down = result_down * delta_energy / (2.0 * M_PI);
			MPI_Reduce(&result_up, &result_reduced_up, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&result_down, &result_reduced_down, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 		} else  {//spin_down = spin_up
			for (int r = 0; r < parameters.steps_myid; r++) {
				result_up += gf_local_lesser_up.at(r)(i, i).imag();
			}
			result_up = result_up * delta_energy / (2.0 * M_PI);
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
 	std::vector<dcomp> &dos_up_metal, std::vector<dcomp> &dos_down_metal, std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &gf_local_down) {
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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
	
	Parameters parameters = Parameters::from_file();
	MPI_Comm_size(MPI_COMM_WORLD, &parameters.size);
	MPI_Comm_rank(MPI_COMM_WORLD, &parameters.myid);
	parameters.comm = MPI_COMM_WORLD;

	if (parameters.myid == 0) {
		print_parameters(parameters);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (parameters.myid == 0) {
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;
	}
	
	parameters.steps_proc.resize(parameters.size, 0);
	parameters.end.resize(parameters.size, 0);
	parameters.start.resize(parameters.size, 0);
	
	decomp(parameters.steps, parameters.size, parameters.myid, &parameters.start.at(parameters.myid), &parameters.end.at(parameters.myid));
	parameters.steps_myid = parameters.end.at(parameters.myid) - parameters.start.at(parameters.myid) + 1;

	std::cout << std::setprecision(15) << "My myid is " << parameters.myid << " in a world of size " << parameters.size << 
		". There are " << parameters.steps<< " energy steps in my parameters class." 
		" The starting point and end point of my array are " << parameters.start.at(parameters.myid) << " and " << parameters.end.at(parameters.myid) << 
		". The number of points in my process are " << parameters.steps_myid << "\n";
		
	parameters.steps_proc.at(parameters.myid) = parameters.steps_myid;

	if (parameters.myid == 0) {
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;
	}

	for (int a = 0; a < parameters.size; a++){
		MPI_Bcast(&parameters.start.at(a), 1, MPI_INT, a, MPI_COMM_WORLD);
		MPI_Bcast(&parameters.end.at(a), 1, MPI_INT, a, MPI_COMM_WORLD);
		MPI_Bcast(&parameters.steps_proc.at(a), 1, MPI_INT, a, MPI_COMM_WORLD);
	}

	//if (parameters.myid == 0) {
	//	for (int i = 0; i < parameters.size; i++){
	//		std::cout << "The number of starting index of " << i << " is " << parameters.start.at(i) << std::endl;
	//		std::cout << "The number of ending index of " << i << " is " << parameters.end.at(i) << std::endl;
	//		std::cout << "The number of steps_proc index of " << i << " is " << parameters.steps_proc.at(i) << std::endl;
	//	}
	//}


	std::vector<double> kx(parameters.num_kx_points, 0);
	std::vector<double> ky(parameters.num_ky_points, 0);
	get_momentum_vectors(kx, ky, parameters);

	std::vector<double> current_up(parameters.NIV_points, 0), current_down(parameters.NIV_points, 0), coherent_current_up(parameters.NIV_points, 0),
		 coherent_current_down(parameters.NIV_points, 0), noncoherent_current_up(parameters.NIV_points, 0), noncoherent_current_down(parameters.NIV_points, 0),
		 current_up_right(parameters.NIV_points, 0), current_up_left(parameters.NIV_points, 0), current_down_right(parameters.NIV_points, 0)
		 , current_down_left(parameters.NIV_points, 0);
	
	MPI_Barrier(MPI_COMM_WORLD); //this is just so the code prints nicely to the out file.

	for (int m = parameters.NIV_start; m < parameters.NIV_points; m++) {
		std::cout << "\n";
		if (parameters.myid == 0) {
			std::cout << std::setprecision(15) << "The voltage difference is " << parameters.voltage_l[m] - parameters.voltage_r[m] << std::endl;
			std::cout << "intialising hamiltonian \n";
			std::cout << "The number of orbitals we have is " << 4 * parameters.chain_length << "\n";
		}

		std::vector<std::vector<Eigen::MatrixXcd>> hamiltonian(
		    parameters.num_kx_points, std::vector<Eigen::MatrixXcd>(parameters.num_ky_points,
				 Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length)));

		std::vector<Eigen::MatrixXcd> gf_local_up(parameters.steps_myid, Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
		std::vector<Eigen::MatrixXcd> gf_local_down(parameters.steps_myid, Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
		std::vector<Eigen::MatrixXcd> gf_local_lesser_up(parameters.steps_myid, Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
		std::vector<Eigen::MatrixXcd> gf_local_lesser_down(parameters.steps_myid, Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
		std::vector<std::vector<dcomp>> self_energy_mb_up(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid)),
		    self_energy_mb_lesser_up(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid, 0));
		std::vector<std::vector<dcomp>> self_energy_mb_down(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid)),
		    self_energy_mb_lesser_down(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid, 0));
		std::vector<double> spins_occup(8 * parameters.chain_length); //the first 2 * chain_length is the spin up, the next 2 * chain_length is spin down.

		std::vector<std::vector<EmbeddingSelfEnergy>> leads;
		for (int i = 0; i < parameters.num_kx_points; i++) {
			std::vector<EmbeddingSelfEnergy> vy;
			for (int j = 0; j < parameters.num_ky_points; j++) {
				vy.push_back(EmbeddingSelfEnergy(parameters, kx.at(i), ky.at(j), m));
			}
			leads.push_back(vy);
		}

		if (parameters.myid == 0) {
			std::cout << "leads complete" << std::endl;
			std::cout << "leads size: " << leads.at(0).size() << '\n';
		}
		//get_k_averaged_embedding_self_energy(parameters, leads);

		for (int kx_i = 0; kx_i < parameters.num_kx_points; kx_i++) {
			for (int ky_i = 0; ky_i < parameters.num_ky_points; ky_i++) {
				get_hamiltonian(parameters, m, kx.at(kx_i), ky.at(ky_i), hamiltonian.at(kx_i).at(ky_i));

				//if (parameters.myid == 0){
				//	std::cout << hamiltonian.at(kx_i).at(ky_i).real() << std::endl;
				//	std::cout << std::endl;					
				//}
				//std::cout << "The hamiltonian on myid " << parameters.myid << " is " <<  std::endl;
				//std::cout << hamiltonian.at(kx_i).at(ky_i) << std::endl;
				//std::cout << std::endl;
			}
		}

		if (parameters.myid == 0) {
			std::cout << "hamiltonian complete" << std::endl;
		}
		//get_spectral_embedding_self_energy(parameters, leads, m);

		if (parameters.hubbard_interaction != 0) {
			if (parameters.spin_polarised == true) {	
				set_initial_spin(parameters, self_energy_mb_up, self_energy_mb_down);

			if (parameters.noneq_test == true) { //this calculates g_lesser via the FD (not correct theoretically)
				get_noneq_test(parameters, self_energy_mb_up, leads, gf_local_up, gf_local_lesser_up, m, hamiltonian);
				get_noneq_test(parameters, self_energy_mb_down, leads, gf_local_down, gf_local_lesser_down, m, hamiltonian);				
			} else {
				get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, m, hamiltonian);
				get_local_gf_r_and_lesser(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, gf_local_down, gf_local_lesser_down, m, hamiltonian);			
			}
		
			} else { //spin up and down are degenerate. Hence eonly need to do this once
				if (parameters.noneq_test == true) { //this calculates g_lesser via the FD (not correct theoretically)
					get_noneq_test(parameters, self_energy_mb_up, leads, gf_local_up, gf_local_lesser_up, m, hamiltonian);
				} else {
					get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, m, hamiltonian);
				}
			}

			if (parameters.myid == 0) {
				std::cout << "got local retarded and lesser gf" << std::endl;
			}

			dmft(parameters, m, self_energy_mb_up, self_energy_mb_down, self_energy_mb_lesser_up, self_energy_mb_lesser_down, gf_local_up, gf_local_down, gf_local_lesser_up,
			    gf_local_lesser_down, leads, spins_occup, hamiltonian);

			if (parameters.myid == 0) {
				std::cout << "got self energy " << std::endl;
			}
		}


		double current_up_myid = 0.0, current_down_myid = 0.0, current_up_left_myid = 0.0, current_up_right_myid = 0.0,
			current_down_left_myid = 0.0, current_down_right_myid = 0.0, coherent_current_up_myid = 0.0, coherent_current_down_myid = 0.0;
			//current_noninteracting_up_myid = 0.0, current_noninteracting_down_myid = 0.0;

		std::vector<dcomp> transmission_up(parameters.steps_myid, 0);
		std::vector<dcomp> transmission_down(parameters.steps_myid, 0);
		//std::vector<double> current_noninteracting_up(parameters.NIV_points, 0), current_noninteracting_down(parameters.NIV_points, 0);
		//std::vector<dcomp> transmission_noninteracting_up(parameters.steps_myid, 0);
		//std::vector<dcomp> transmission_noninteracting_down(parameters.steps_myid, 0);
		if (parameters.hubbard_interaction == 0) {
			get_transmission_gf_local(parameters, self_energy_mb_up, self_energy_mb_down, leads, transmission_up, transmission_down,
    			 m, hamiltonian, gf_local_up, gf_local_lesser_up);
			if (parameters.myid == 0) {			
				std::cout << "got transmission\n";
			}
			get_landauer_buttiker_current(parameters, transmission_up, transmission_down, &current_up_myid, &current_down_myid, m);

			MPI_Reduce(&current_up_myid, &current_up.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&current_down_myid, &current_down.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);	

			if (parameters.myid == 0) {
				std::cout << "The spin up current is " << current_up.at(m) << "\n" <<
					 "The spin down current is " << current_down.at(m) << "\n" << "\n";	
			}
	
		} else {
			if (parameters.spin_polarised == true) {
				get_meir_wingreen_current(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m, &current_up_left_myid, &current_up_right_myid, 
					transmission_up, hamiltonian);
				get_meir_wingreen_current(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, m, &current_down_left_myid, &current_down_right_myid,
					transmission_down, hamiltonian);
				get_landauer_buttiker_current(parameters, transmission_up, transmission_down, &coherent_current_up_myid, &coherent_current_down_myid, m);
			} else { //spin down =spin_up
				get_meir_wingreen_current(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m, &current_up_left_myid, &current_up_right_myid, 
					transmission_up, hamiltonian);
				for(int r = 0; r < parameters.steps_myid; r++) {
					transmission_down.at(r) = transmission_up.at(r);
				}
				current_down_left_myid = current_up_left_myid;
				current_down_right_myid = current_up_right_myid;
				get_landauer_buttiker_current(parameters, transmission_up, transmission_down, &coherent_current_up_myid, &coherent_current_down_myid, m);
			}
		}


		MPI_Reduce(&current_up_right_myid, &current_up_right.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&current_up_left_myid, &current_up_left.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&current_down_right_myid, &current_down_right.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&current_down_left_myid, &current_down_left.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&coherent_current_up_myid, &coherent_current_up.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&coherent_current_down_myid, &coherent_current_down.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		//MPI_Reduce(&current_noninteracting_up_myid, &current_noninteracting_up.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		
		if (parameters.myid == 0) {
			noncoherent_current_up.at(m) = 0.5 * (current_up_left.at(m) - current_up_right.at(m)) - coherent_current_up.at(m);
			noncoherent_current_down.at(m) = 0.5 * (current_down_left.at(m) - current_down_right.at(m)) - coherent_current_down.at(m);
			std::cout << "The spin up left current is " << current_up_left.at(m) << "\n" <<
					 "The spin up right current is " << current_up_right.at(m) << "\n" <<
					 "The spin down left current is " << current_down_left.at(m) << "\n" <<
					 "The spin up right current is " << current_down_right.at(m) << "\n" <<
					 "The total current is " << 0.5 * (current_down_left.at(m) - current_down_right.at(m)) << "\n" <<
					 "The coherent current is " << coherent_current_down.at(m) << "\n" <<
					 "The noncoherent current is " << noncoherent_current_down.at(m) << "\n"; 
					 //"The noninteracting current is " << current_noninteracting_up.at(m) << "\n";
			std::cout << std::endl;
		}

		//for (int i = 0; i < 4 * parameters.chain_length; i++) {
		//	if (parameters.atom_type.at(i) != 0) {
		//		if (parameters.myid == 0) {
		//			std::cout << "The spin up occupation at atom " << i << " is " << spins_occup.at(i) << std::endl;
		//			std::cout << "The spin down occupation at atom " << i << " is " << spins_occup.at(i + 4 * parameters.chain_length) << std::endl;
		//		}				
		//	}
		//}
//
		std::vector<dcomp> dos_up(parameters.steps_myid, 0);
		std::vector<dcomp> dos_down(parameters.steps_myid, 0);
		std::vector<dcomp> dos_up_ins(parameters.steps_myid, 0);
		std::vector<dcomp> dos_down_ins(parameters.steps_myid, 0);
		std::vector<dcomp> dos_up_metal(parameters.steps_myid, 0);
		std::vector<dcomp> dos_down_metal(parameters.steps_myid, 0);	
		if (parameters.hubbard_interaction == 0) {
			get_dos(parameters, dos_up, dos_down, dos_up_ins, dos_down_ins, dos_up_metal, dos_down_metal, gf_local_up, gf_local_up);

			//spin up and down are degenerate
		} else {
			if (parameters.spin_polarised == true) {
				get_dos(parameters, dos_up, dos_down, dos_up_ins, dos_down_ins, dos_up_metal, dos_down_metal, gf_local_up, gf_local_down);
			} else {
				get_dos(parameters, dos_up, dos_down, dos_up_ins, dos_down_ins, dos_up_metal, dos_down_metal, gf_local_up, gf_local_up);
			}
		}
		
		get_occupation(parameters, gf_local_lesser_up, gf_local_lesser_down, spins_occup);

		if (parameters.myid == 0) {
			std::cout << parameters.print_gf << std::endl;
		}
		if (parameters.print_gf == true) {
			if (parameters.spin_polarised == true) {
				write_to_file(parameters, gf_local_up, gf_local_down, "gf.txt", m);
				write_to_file(parameters, gf_local_lesser_up, gf_local_lesser_down, "gf_lesser.txt", m);
			} else {
				write_to_file(parameters, gf_local_up, gf_local_up, "gf.txt", m);
				write_to_file(parameters, gf_local_lesser_up, gf_local_lesser_up, "gf_lesser.txt", m);
			}
		}

		write_to_file(parameters, transmission_up, transmission_down, "transmission.txt", m);
		//write_to_file(parameters, transmission_noninteracting_up, transmission_noninteracting_down, "transmission_noninteracting.txt", m);
		write_to_file(parameters, dos_up, dos_down, "dos.txt", m);
		write_to_file(parameters, dos_up_ins, dos_down_ins, "dos_ins.txt", m);
		write_to_file(parameters, dos_up_metal, dos_down_metal, "dos_metal.txt", m);
		if (parameters.spin_polarised == true) {
			write_to_file(parameters, self_energy_mb_up, self_energy_mb_down, "se_r.txt", m);
			write_to_file(parameters, self_energy_mb_lesser_up, self_energy_mb_lesser_down, "se_l.txt", m);
		} else {
			write_to_file(parameters, self_energy_mb_up, self_energy_mb_up, "se_r.txt", m);
			write_to_file(parameters, self_energy_mb_lesser_up, self_energy_mb_lesser_up, "se_l.txt", m);
		}

		integrate_spectral(parameters, gf_local_up);

		if (parameters.myid == 0) {			
			std::cout << "wrote files\n";
		}

		if (parameters.num_kx_points == 1 && parameters.num_ky_points == 1 && parameters.chain_length == 1 && parameters.hubbard_interaction == 0){
			analytic_gf(parameters, gf_local_up);
			if (parameters.myid == 0) {			
				std::cout << "got analytic gf\n";
			}
		}
	}

	if (parameters.myid == 0) {
		if (parameters.hubbard_interaction == 0) {
			std::ofstream current_file;
			current_file.open(
			    "textfiles/"
			    "current.txt");
			// myfile << parameters.steps << std::endl;
			for (int m = 0; m < parameters.NIV_points; m++) {
				//std::cout << "The spin up current is " << current_up.at(m) << "The spin down current is " << current_down.at(m) << "\n";

				if (parameters.hubbard_interaction == 0.0) {
					std::cout << "The spin up current is " << current_up.at(m) << 
					"\n The spin down current is " << current_down.at(m) << "\n";
				}

				std::cout << "\n";
				current_file << parameters.voltage_l[m] - parameters.voltage_r[m]
				             << "   "
				             //<< current_up.at(m).real() << "   "
				             //<< current_down.at(m).real() << "   "
				             << 2 * current_up_left.at(m) << "   " << 2 * current_up_right.at(m)
				             << "   "
				             //<< current_down_left.at(m).real() << "   "
				             << 2 * current_up_left.at(m) + 2 * current_up_right.at(m) << "\n";
			}
			current_file.close();
		} else {
			std::ofstream current_file;
			current_file.open(
			    "textfiles/"
			    "current.txt");
			for (int m = 0; m < parameters.NIV_points; m++) {

				std::cout << "The spin up left current is " << current_up_left.at(m) << "\n" <<
						 "The spin up right current is " << current_up_right.at(m) << "\n" <<
						 "The spin down left current is " << current_down_left.at(m) << "\n" <<
						 "The spin up right current is " << current_down_right.at(m) << "\n" <<
						 "The total current is " << 0.5 * (current_down_left.at(m) - current_down_right.at(m)) << "\n" <<
						 "The coherent current is " << coherent_current_down.at(m) << "\n" <<
						 "The noncoherent current is " << noncoherent_current_down.at(m) << "\n";

				std::cout << "\n";

				current_file << parameters.voltage_l[m] - parameters.voltage_r[m] << "   " << current_up_left.at(m) << "   " << current_up_right.at(m) << "   "
				             << current_down_left.at(m) << "   " << current_down_right.at(m) << "     " << 0.5 * (current_down_left.at(m) - current_down_right.at(m))
				             << "  " << coherent_current_down.at(m) << "   " << noncoherent_current_down.at(m) << "\n";
			}
			current_file.close();
		}
	}
	MPI_Finalize();
	return 0;
}
