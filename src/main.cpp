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

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	Parameters parameters = Parameters::from_file();
	std::vector<double> kx(parameters.num_kx_points, 0), ky(parameters.num_ky_points, 0);
	get_momentum_vectors(kx, ky, parameters);

	std::vector<double> coherent_current_up(parameters.NIV_points, 0);
	std::vector<double> coherent_current_down(parameters.NIV_points, 0);
	std::vector<double> noncoherent_current_up(parameters.NIV_points, 0);
	std::vector<double> noncoherent_current_down(parameters.NIV_points, 0);
	std::vector<double> current_up_right(parameters.NIV_points, 0);
	std::vector<double> current_up_left(parameters.NIV_points, 0);
	std::vector<double> current_down_right(parameters.NIV_points, 0);
	std::vector<double> current_down_left(parameters.NIV_points, 0);
	std::vector<double> bond_current_up(parameters.NIV_points, 0);
	std::vector<double> bond_current_down(parameters.NIV_points, 0);


	for (int m = parameters.NIV_start; m < parameters.NIV_points; m++) {
		

		if (parameters.myid == 0) {
			std::cout << std::setprecision(15) << "\n The voltage difference is " << parameters.voltage_l[m] - parameters.voltage_r[m] << std::endl;
			std::cout << "intialising hamiltonian \n";
			std::cout << "The number of orbitals we have is " << 4 * parameters.chain_length << "\n";
		}

		//the unit cell is 2x2xchain_length. The 2x2 is required for the insulating layers. hence all quantities in the central region are matrices
		//of size 4* parameters.chain_length.

		//this creates the k-depedent hamiltonian for both spins
		std::vector<MatrixVectorType> hamiltonian_up(parameters.num_kx_points, MatrixVectorType(parameters.num_ky_points,
			MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length)));
		std::vector<MatrixVectorType> hamiltonian_down(parameters.num_kx_points, MatrixVectorType(parameters.num_ky_points,
			MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length)));
		
		for (int kx_i = 0; kx_i < parameters.num_kx_points; kx_i++) {
			for (int ky_i = 0; ky_i < parameters.num_ky_points; ky_i++) {//1 refers to spin up. 2 refers to spin down
				get_hamiltonian(parameters, m, kx.at(kx_i), ky.at(ky_i), hamiltonian_up.at(kx_i).at(ky_i), 1); 
				get_hamiltonian(parameters, m, kx.at(kx_i), ky.at(ky_i), hamiltonian_down.at(kx_i).at(ky_i), 2); 
			}
		}
		
		if (parameters.myid == 0) std::cout << "hamiltonian complete" << std::endl;

		std::vector<std::vector<EmbeddingSelfEnergy>> leads;
		for (int i = 0; i < parameters.num_kx_points; i++) {
			std::vector<EmbeddingSelfEnergy> vy;
			for (int j = 0; j < parameters.num_ky_points; j++) {
				vy.push_back(EmbeddingSelfEnergy(parameters, kx.at(i), ky.at(j), m));
			}
			leads.push_back(vy);
		}

		if (parameters.myid == 0) std::cout << "leads complete" << std::endl;

		MatrixVectorType gf_local_up(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
		MatrixVectorType gf_local_down(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
		MatrixVectorType gf_local_lesser_up(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
		MatrixVectorType gf_local_lesser_down(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));

		MatrixVectorType gf_local_greater_up, gf_local_greater_down;
		std::vector<std::vector<dcomp>> self_energy_mb_greater_down, self_energy_mb_greater_up;

		if (parameters.impurity_solver == 3) {//only bother to intiliase the greater stuff if we are doing the nca loop.
			gf_local_greater_up.resize(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
			gf_local_greater_down.resize(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
			self_energy_mb_greater_down.resize(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid));
		    self_energy_mb_greater_up.resize(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid, 0));
		}

		std::vector<std::vector<dcomp>> self_energy_mb_up(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid));
		std::vector<std::vector<dcomp>> self_energy_mb_lesser_up(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid, 0));
		std::vector<std::vector<dcomp>>	self_energy_mb_down(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid));
		std::vector<std::vector<dcomp>> self_energy_mb_lesser_down(4 * parameters.chain_length, std::vector<dcomp>(parameters.steps_myid, 0));
		
		std::vector<double> spins_occup(8 * parameters.chain_length); //the first 2 * chain_length is the spin up, the next 2 * chain_length is spin down.

		double current_up_left_myid = 0.0, current_up_right_myid = 0.0,
			current_down_left_myid = 0.0, current_down_right_myid = 0.0, coherent_current_up_myid = 0.0, coherent_current_down_myid = 0.0;
				//current_noninteracting_up_myid = 0.0, current_noninteracting_down_myid = 0.0;

		std::vector<dcomp> transmission_up(parameters.steps_myid, 0), transmission_down(parameters.steps_myid, 0);

		if (parameters.hubbard_interaction == 0) {
			if (parameters.meir_wingreen_current == 1) {
				//this gets the meir-wingreen current
				//we calculate the local gf and trc at the same time
				//the self energies are all zero.
				get_transmission_gf_local(parameters, self_energy_mb_up, self_energy_mb_down, leads, transmission_up, transmission_down,
    				 m, hamiltonian_up, hamiltonian_down, gf_local_up, gf_local_lesser_up, gf_local_down, gf_local_lesser_down);

				if (parameters.myid == 0) std::cout << "got transmission\n";

				get_meir_wingreen_current(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m, &current_up_left_myid, &current_up_right_myid, 
					transmission_up, hamiltonian_up);
				get_meir_wingreen_current(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, m, &current_down_left_myid, &current_down_right_myid,
					transmission_down, hamiltonian_down);
				get_landauer_buttiker_current(parameters, transmission_up, transmission_down, &coherent_current_up_myid, &coherent_current_down_myid, m);
				//get_landauer_buttiker_current(parameters, transmission_up, transmission_down, &current_up_myid, &current_down_myid, m);
			}

			if (parameters.bond_current == 1) {
				if (parameters.spin_polarised == true) {
					if (parameters.myid == 0) std::cout << "Calulating the bond currents for a spin polarised calculation \n";
					get_bond_current(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m,  &bond_current_up.at(m), hamiltonian_up);
					get_bond_current(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, m,  &bond_current_down.at(m), hamiltonian_down);
				} else {//spin up = spin down
					if (parameters.myid == 0) std::cout << "Calulating the bond currents for a non spin polarised calculation \n";
					get_bond_current(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m,  &bond_current_up.at(m), hamiltonian_up);
					bond_current_down.at(m) = bond_current_up.at(m);
				}
			}
		} else if (parameters.hubbard_interaction != 0) {
			if (parameters.spin_polarised == true) {	
				set_initial_spin(parameters, self_energy_mb_up, self_energy_mb_down);
				if (parameters.impurity_solver == 3) {//this calculates g^> as well which is required for the nca.
					get_local_gf_r_greater_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, self_energy_mb_greater_up, leads, gf_local_up, gf_local_lesser_up,
					 	gf_local_greater_up, m, hamiltonian_up);
					get_local_gf_r_greater_lesser(parameters, self_energy_mb_down, self_energy_mb_lesser_down, self_energy_mb_greater_down, leads,
						 gf_local_down, gf_local_lesser_down, gf_local_greater_down, m, hamiltonian_down);	
				} else {//only need gf_retarded and gf_lesser for sigma2
					get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, m, hamiltonian_up);
					get_local_gf_r_and_lesser(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, gf_local_down, gf_local_lesser_down, m, hamiltonian_down);	
				}

			} else { //spin up and down are degenerate. Hence only need to do this once. choose to do it for spin up
				if (parameters.impurity_solver == 3) {//this calculates g^> as well which is required for the nca.
					get_local_gf_r_greater_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, self_energy_mb_greater_up, leads, gf_local_up, gf_local_lesser_up,
				 		gf_local_greater_up, m, hamiltonian_up);
				} else {//only need gf_retarded and gf_lesser for sigma2
					get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, m, hamiltonian_up);
				}
			}

			if (parameters.myid == 0) std::cout << "got local retarded and lesser gf" << std::endl;

			dmft(parameters, m, self_energy_mb_up, self_energy_mb_down, self_energy_mb_lesser_up, self_energy_mb_lesser_down,
				self_energy_mb_greater_up, self_energy_mb_greater_down, gf_local_up, gf_local_down, gf_local_lesser_up, gf_local_lesser_down,
				gf_local_greater_up, gf_local_greater_down, leads, spins_occup, hamiltonian_up, hamiltonian_down);
			if (parameters.myid == 0) std::cout << "got self energy " << std::endl;

			if (parameters.spin_polarised == true) {//the transmission is calculated while calculating the MW for computational speed up
				//non spin degerneate.
				get_meir_wingreen_current(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m, &current_up_left_myid, &current_up_right_myid, 
					transmission_up, hamiltonian_up);
				get_meir_wingreen_current(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, m, &current_down_left_myid, &current_down_right_myid,
					transmission_down, hamiltonian_down);
				get_landauer_buttiker_current(parameters, transmission_up, transmission_down, &coherent_current_up_myid, &coherent_current_down_myid, m);
			} else { //spin down =spin_up. if spin polarise dis not true then there is no magnetic field by the code in parameters.cpp
				get_meir_wingreen_current(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m, &current_up_left_myid, &current_up_right_myid, 
					transmission_up, hamiltonian_up);
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
			if (parameters.bond_current == true) {
				std::cout << "The spin up bond current is " << bond_current_up.at(m) << "\n" <<
					 "The spin up bond current is " << bond_current_down.at(m) << "\n";
			}
			std::cout << std::endl;
		}
			
		std::vector<dcomp> dos_up(parameters.steps_myid, 0), dos_down(parameters.steps_myid, 0);
		std::vector<dcomp> dos_up_ins(parameters.steps_myid, 0), dos_down_ins(parameters.steps_myid, 0);
		std::vector<dcomp> dos_up_metal(parameters.steps_myid, 0), dos_down_metal(parameters.steps_myid, 0);	
		
		if (parameters.spin_polarised == true) {
			get_dos(parameters, dos_up, dos_down, dos_up_ins, dos_down_ins, dos_up_metal, dos_down_metal, gf_local_up, gf_local_down);
		} else {
			get_dos(parameters, dos_up, dos_down, dos_up_ins, dos_down_ins, dos_up_metal, dos_down_metal, gf_local_up, gf_local_up);
		}
		
		get_occupation(parameters, gf_local_lesser_up, gf_local_lesser_down, spins_occup);
		
		if (parameters.myid == 0) {
			std::cout << parameters.print_gf << std::endl;
		}
		
		if (parameters.print_gf == true) {//this is code to print the local gf functions
			if (parameters.spin_polarised == true) {
				std::cout << "here \n";
				write_to_file(parameters, gf_local_up, gf_local_down, "gf.dat", m);
				write_to_file(parameters, gf_local_lesser_up, gf_local_lesser_down, "gf_lesser.dat", m);
				if (parameters.impurity_solver == 3) {
					write_to_file(parameters, gf_local_greater_up, gf_local_greater_down, "gf_greater.dat", m);
				}
			} else {//we havent calculated spin down gf so we just pass the spin up one twice.
				write_to_file(parameters, gf_local_up, gf_local_up, "gf.dat", m);
				write_to_file(parameters, gf_local_lesser_up, gf_local_lesser_up, "gf_lesser.dat", m);
				if (parameters.impurity_solver == 3) {
					write_to_file(parameters, gf_local_greater_up, gf_local_greater_up, "gf_greater.dat", m);
				}
			}
		}
		
		write_to_file(parameters, transmission_up, transmission_down, "transmission.dat", m);
		write_to_file(parameters, dos_up, dos_down, "dos.dat", m);
		write_to_file(parameters, dos_up_ins, dos_down_ins, "dos_ins.dat", m);
		write_to_file(parameters, dos_up_metal, dos_down_metal, "dos_metal.dat", m);
		
		if (parameters.hubbard_interaction != 0) {
			if (parameters.spin_polarised == true) {
				write_to_file(parameters, self_energy_mb_up, self_energy_mb_down, "se_r.dat", m);
				write_to_file(parameters, self_energy_mb_lesser_up, self_energy_mb_lesser_down, "se_l.dat", m);
			} else {
				write_to_file(parameters, self_energy_mb_up, self_energy_mb_up, "se_r.dat", m);
				write_to_file(parameters, self_energy_mb_lesser_up, self_energy_mb_lesser_up, "se_l.dat", m);
			}
		}
		
		integrate_spectral(parameters, gf_local_up);
		
		if (parameters.myid == 0) std::cout << "wrote files\n";
	}

	if (parameters.myid == 0) {
		std::ofstream current_file;
		current_file.open("current.dat");
		for (int m = 0; m < parameters.NIV_points; m++) {
			noncoherent_current_down.at(m) = 0.5 * (current_down_left.at(m) - current_down_right.at(m)) - coherent_current_down.at(m);
			std::cout << "The spin up left current is " << current_up_left.at(m) << "\n" <<
				"The spin up right current is " << current_up_right.at(m) << "\n" <<
				"The spin down left current is " << current_down_left.at(m) << "\n" <<
				"The spin up right current is " << current_down_right.at(m) << "\n" <<
				"The total spin down current is " << 0.5 * (current_down_left.at(m) - current_down_right.at(m)) << "\n" <<
				"The spin down coherent current is " << coherent_current_down.at(m) << "\n" <<
				"The spin down noncoherent current is " << noncoherent_current_down.at(m) << "\n";
			current_file << parameters.voltage_l[m] - parameters.voltage_r[m] << "   " << current_up_left.at(m) << "   " << current_up_right.at(m) << "   "
			             << current_down_left.at(m) << "   " << current_down_right.at(m) << "     " << 0.5 * (current_down_left.at(m) - current_down_right.at(m))
			             << "  " << coherent_current_down.at(m) << "   " << noncoherent_current_down.at(m);
			if (parameters.bond_current == 1) {
				current_file << "   " << bond_current_up.at(m) << "   " << bond_current_down.at(m) << "\n";
				std::cout <<				"The spin up bond currentis " << bond_current_up.at(m) << "\n" <<
				"The spin down bond currentis " << bond_current_down.at(m) << "\n" ;
			} else {
				current_file << "\n";
			}
			std::cout << "\n";
		}
		current_file.close();
	}
	MPI_Finalize();
	return 0;
}