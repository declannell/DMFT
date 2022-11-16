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
	if (parameters.ins_metal_ins == true){
		for (int i = parameters.num_ins_left; i < parameters.num_cor + parameters.num_ins_left; i++) {
			for (int r = 0; r < parameters.steps; r++) {
				self_energy_mb_up.at(i).at(r) = parameters.spin_down_occup * parameters.hubbard_interaction;
				self_energy_mb_down.at(i).at(r) = parameters.spin_up_occup * parameters.hubbard_interaction;
				self_energy_mb_up.at(i + parameters.chain_length).at(r) = parameters.spin_down_occup * parameters.hubbard_interaction;//this is for the second layer in unit cell
				self_energy_mb_down.at(i + parameters.chain_length).at(r) = parameters.spin_up_occup * parameters.hubbard_interaction;//this is for the second layer in unit cell
			}
		}
	} else { //this is for the metal-ins-metal junction. These are the atoms on the left side of the insulator
		for(int i = 0; i < parameters.num_cor; i++) {
			for (int r = 0; r < parameters.steps; r++) {
				self_energy_mb_up.at(i).at(r) = parameters.spin_down_occup * parameters.hubbard_interaction;
				self_energy_mb_down.at(i).at(r) = parameters.spin_up_occup * parameters.hubbard_interaction;
				self_energy_mb_up.at(i + parameters.chain_length).at(r) = parameters.spin_down_occup * parameters.hubbard_interaction;
				self_energy_mb_down.at(i + parameters.chain_length).at(r) = parameters.spin_up_occup * parameters.hubbard_interaction;
			}
		}
		//These are the atoms on the left side of the insulator
		for(int i = parameters.num_cor + parameters.num_ins_left; i < parameters.chain_length; i++) {
			for (int r = 0; r < parameters.steps; r++) {
				self_energy_mb_up.at(i).at(r) = parameters.spin_down_occup * parameters.hubbard_interaction;
				self_energy_mb_down.at(i).at(r) = parameters.spin_up_occup * parameters.hubbard_interaction;
				self_energy_mb_up.at(i + parameters.chain_length).at(r) = parameters.spin_down_occup * parameters.hubbard_interaction;
				self_energy_mb_down.at(i + parameters.chain_length).at(r) = parameters.spin_up_occup * parameters.hubbard_interaction;
			}			
		}
	}
}

void integrate_spectral(Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local){
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
	for(int i = 0; i < 2 * parameters.chain_length; i++){
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

void get_dos(Parameters &parameters, std::vector<dcomp> &dos_up, std::vector<dcomp> &dos_down,  std::vector<Eigen::MatrixXcd> &gf_local_up, 
	 std::vector<Eigen::MatrixXcd> &gf_local_down) {
		for (int r = 0; r < parameters.steps_myid; r++) {
			for (int i = 0; i < 2 * parameters.chain_length; i++) {
				dos_up.at(r) += -gf_local_up.at(r)(i, i).imag();
				dos_down.at(r) += -gf_local_down.at(r)(i, i).imag();
			}
		}
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	Parameters parameters = Parameters::init();
	
	if (parameters.myid == 0) {
		print_parameters(parameters);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &parameters.size);
	MPI_Comm_rank(MPI_COMM_WORLD, &parameters.myid);

	std::cout << "My myid is " << parameters.myid << std::endl;

	parameters.steps_proc.resize(parameters.size, 0);
	parameters.end.resize(parameters.size, 0);
	parameters.start.resize(parameters.size, 0);
	
	decomp(parameters.steps, parameters.size, parameters.myid, &parameters.start.at(parameters.myid), &parameters.end.at(parameters.myid));
	parameters.steps_myid = parameters.end.at(parameters.myid) - parameters.start.at(parameters.myid) + 1;

	std::cout << "My myid is " << parameters.myid << " in a world of size " << parameters.size << 
		". There are " << parameters.steps<< " energy steps in my parameters class." 
		" The starting point and end point of my array are " << parameters.start.at(parameters.myid) << " and " << parameters.end.at(parameters.myid) << 
		". The number of points in my process are " << parameters.steps_myid << "\n";

	parameters.steps_proc.at(parameters.myid) = parameters.steps_myid;

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

	for (int m = 1; m < parameters.NIV_points; m++) {
		std::cout << "\n";
		if (parameters.myid == 0) {
			std::cout << std::setprecision(15) << "The voltage difference is " << parameters.voltage_l[m] - parameters.voltage_r[m] << std::endl;
			std::cout << "intialising hamiltonian \n";
		}

		std::vector<std::vector<Eigen::MatrixXcd>> hamiltonian(
		    parameters.num_kx_points, std::vector<Eigen::MatrixXcd>(parameters.num_ky_points,
				 Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length)));

		std::vector<Eigen::MatrixXcd> gf_local_up(parameters.steps_myid, Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
		std::vector<Eigen::MatrixXcd> gf_local_down(parameters.steps_myid, Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
		std::vector<Eigen::MatrixXcd> gf_local_lesser_up(parameters.steps_myid, Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
		std::vector<Eigen::MatrixXcd> gf_local_lesser_down(parameters.steps_myid, Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
		std::vector<std::vector<dcomp>> self_energy_mb_up(2 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid)),
		    self_energy_mb_lesser_up(2 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid, 0));
		std::vector<std::vector<dcomp>> self_energy_mb_down(2 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid)),
		    self_energy_mb_lesser_down(2 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid, 0));

		std::vector<std::vector<EmbeddingSelfEnergy>> leads;
		for (int i = 0; i < parameters.num_kx_points; i++) {
			std::vector<EmbeddingSelfEnergy> vy;
			for (int j = 0; j < parameters.num_ky_points; j++) {
				vy.push_back(EmbeddingSelfEnergy(parameters, kx.at(i), ky.at(j), m));
			}
			leads.push_back(vy);
		}

		if (parameters.myid == 0) {
			std::cout << "leads size: " << leads.at(0).size() << '\n';
		}
		//get_k_averaged_embedding_self_energy(parameters, leads);

		for (int kx_i = 0; kx_i < parameters.num_kx_points; kx_i++) {
			for (int ky_i = 0; ky_i < parameters.num_ky_points; ky_i++) {
				get_hamiltonian(parameters, m, kx.at(kx_i), ky.at(ky_i), hamiltonian.at(kx_i).at(ky_i));
				//std::cout << "The hamiltonian on myid " << parameters.myid << " is " <<  std::endl;
				//std::cout << hamiltonian.at(kx_i).at(ky_i) << std::endl;
				//std::cout << std::endl;
			}
		}
		//get_spectral_embedding_self_energy(parameters, leads, m);

		if (parameters.myid == 0) {
			std::cout << "leads complete" << std::endl;
		}

		get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, m, hamiltonian);
		get_local_gf_r_and_lesser(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, gf_local_down, gf_local_lesser_down, m, hamiltonian);
		
		if (parameters.myid == 0) {
			std::cout << "got local retarded and lesser gf" << std::endl;
		}

		std::vector<double> spins_occup(4 * parameters.chain_length); //the first 2 * chain_length is the spin up, the next 2 * chain_length is spin down.

		dmft(parameters, m, self_energy_mb_up, self_energy_mb_down, self_energy_mb_lesser_up, self_energy_mb_lesser_down, gf_local_up, gf_local_down, gf_local_lesser_up,
		    gf_local_lesser_down, leads, spins_occup, hamiltonian);

		if (parameters.myid == 0) {
			std::cout << "got self energy " << std::endl;
		}

		if(parameters.hubbard_interaction == 0.0 && parameters.num_kx_points == 1 && parameters.num_ky_points == 1 && 
			parameters.num_ins_left == 0 && parameters.ins_metal_ins == true){
				analytic_gf(parameters, gf_local_up);
		}



		for (int i = 0; i < 2 * parameters.chain_length; i++) {
			if (parameters.atom_type.at(i) != 0) {
				if (parameters.myid == 0) {
				std::cout << "The spin up occupation at atom " << i << " is " << spins_occup.at(i) << std::endl;
				std::cout << "The spin down occupation at atom " << i << " is " << spins_occup.at(i + 2 * parameters.chain_length) << std::endl;
				}				
			}
		}

		std::vector<dcomp> dos_up(parameters.steps_myid, 0);
		std::vector<dcomp> dos_down(parameters.steps_myid, 0);

		get_dos(parameters, dos_up, dos_down, gf_local_up, gf_local_down);

		double current_up_myid = 0.0, current_down_myid = 0.0, current_up_left_myid = 0.0, current_up_right_myid = 0.0,
			current_down_left_myid = 0.0, current_down_right_myid = 0.0, coherent_current_up_myid = 0.0, coherent_current_down_myid = 0.0;

		std::vector<dcomp> transmission_up(parameters.steps_myid, 0);
		std::vector<dcomp> transmission_down(parameters.steps_myid, 0);
		if (parameters.hubbard_interaction == 0) {
			get_transmission(parameters, self_energy_mb_up, self_energy_mb_down, leads, transmission_up, transmission_down, m, hamiltonian);
			std::cout << "got transmission\n";
			get_landauer_buttiker_current(parameters, transmission_up, transmission_down, &current_up_myid, &current_down_myid, m);
			get_meir_wingreen_current(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m, &current_up_left_myid, &current_up_right_myid, hamiltonian);
			get_meir_wingreen_current(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, m, &current_down_left_myid, &current_down_right_myid, hamiltonian);
			
			MPI_Reduce(&current_up_myid, &current_up.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&current_down_myid, &current_down.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);	

			
			if (parameters.myid == 0) {
				std::cout << "The spin up current is " << current_up.at(m) << "\n" <<
					 "The spin down current is " << current_down.at(m) << "\n" << "\n";	
			}
	
		} else {
			get_meir_wingreen_current(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m, &current_up_left_myid, &current_up_right_myid, hamiltonian);
			get_meir_wingreen_current(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, m, &current_down_left_myid, &current_down_right_myid, hamiltonian);
			get_transmission(parameters, self_energy_mb_up, self_energy_mb_down, leads, transmission_up, transmission_down, m, hamiltonian);
			get_landauer_buttiker_current(parameters, transmission_up, transmission_down, &coherent_current_up_myid, &coherent_current_down_myid, m);
		}


		MPI_Reduce(&current_up_right_myid, &current_up_right.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&current_up_left_myid, &current_up_left.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&current_down_right_myid, &current_down_right.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&current_down_left_myid, &current_down_left.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&coherent_current_up_myid, &coherent_current_up.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&coherent_current_down_myid, &coherent_current_down.at(m), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

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
			std::cout << std::endl;
		}

		write_to_file(parameters, gf_local_up, gf_local_down, "gf.txt", m);
		write_to_file(parameters, gf_local_lesser_up, gf_local_lesser_down, "gf_lesser.txt", m);
		write_to_file(parameters, transmission_up, transmission_down, "transmission.txt", m);
		write_to_file(parameters, dos_up, dos_down, "dos.txt", m);
		write_to_file(parameters, self_energy_mb_up, self_energy_mb_down, "se_r.txt", m);
		write_to_file(parameters, self_energy_mb_lesser_up, self_energy_mb_lesser_up, "se_l.txt", m);
		integrate_spectral(parameters, gf_local_up);
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

				std::cout << "The spin up left current is " << current_up_left.at(m) << 
				"\n The spin up right current is " << current_up_right.at(m) << 
				"\n The spin down left current is " << current_down_left.at(m) <<
				 "\n The spin up right current is " << current_down_right.at(m) << "\n";

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