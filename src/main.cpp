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

void print_parameters(Parameters& parameters)
{
	std::cout << " .onsite_cor = " << parameters.onsite_cor << std::endl;
	std::cout << "onsite_ins_l = " << parameters.onsite_ins_l << std::endl;
	std::cout << "onsite_ins_r = " << parameters.onsite_ins_r << std::endl;
	std::cout << "onsite_l = " << parameters.onsite_l << std::endl;
	std::cout << "onsite_r = " << parameters.onsite_r << std::endl;
	std::cout << "hopping_cor = " << parameters.hopping_cor << std::endl;
	std::cout << "hopping_ins_l = " << parameters.hopping_ins_l << std::endl;
	std::cout << "hopping_ins_r = " << parameters.hopping_ins_r << std::endl;
	std::cout << "hopping_y = " << parameters.hopping_y << std::endl;
	std::cout << "hopping_x = " << parameters.hopping_x << std::endl;
	std::cout << "hopping_lz = " << parameters.hopping_lz << std::endl;
	std::cout << "hopping_ly = " << parameters.hopping_ly << std::endl;
	std::cout << "hopping_lx = " << parameters.hopping_lx << std::endl;
	std::cout << "hopping_rz = " << parameters.hopping_rz << std::endl;
	std::cout << "hopping_ry = " << parameters.hopping_ry << std::endl;
	std::cout << "hopping_rx = " << parameters.hopping_rx << std::endl;
	std::cout << "hopping_lc = " << parameters.hopping_lc << std::endl;
	std::cout << "hopping_rc = " << parameters.hopping_rc << std::endl;
	std::cout << "hopping_ins_l_cor = " << parameters.hopping_ins_l_cor << std::endl;
	std::cout << "hopping_ins_r_cor = " << parameters.hopping_ins_r_cor << std::endl;
	std::cout << "num_cor = " << parameters.num_cor << std::endl;
	std::cout << "parameters.num_ins_left  =" << parameters.num_ins_left << std::endl;
	std::cout << "num_ins_right = " << parameters.num_ins_right << std::endl;
	std::cout << "num_ky_points = " << parameters.num_ky_points << std::endl;
	std::cout << "num_kx_points = " << parameters.num_kx_points << std::endl;
	std::cout << "chemical_potential = " << parameters.chemical_potential << std::endl;
	std::cout << "temperature = " << parameters.temperature << std::endl;
	std::cout << "e_upper_bound = " << parameters.e_upper_bound << std::endl;
	std::cout << "e_lower_bound = " << parameters.e_lower_bound << std::endl;
	std::cout << "hubbard_interaction = " << parameters.hubbard_interaction << std::endl;
	std::cout << "voltage_step = " << parameters.voltage_step << std::endl;
	std::cout << "self_consistent_steps = " << parameters.self_consistent_steps << std::endl;
	std::cout << "read_in_self_energy = " << parameters.read_in_self_energy << std::endl;
	std::cout << "NIV_points = " << parameters.NIV_points << std::endl;
	std::cout << "delta_v = " << parameters.delta_v << std::endl;
	std::cout << "delta_leads = " << parameters.delta_leads << std::endl;
	std::cout << "delta_gf = " << parameters.delta_gf << std::endl;
	std::cout << "leads_3d = " << parameters.leads_3d << std::endl;
	std::cout << "parameters.interaction_order = " << parameters.interaction_order << std::endl;
	std::cout << "parameters.steps = " << parameters.steps << std::endl;
	std::cout << "parameters.chain_length = " << parameters.chain_length << std::endl;
	std::cout << "parameters.spin_up_occup = " << parameters.spin_up_occup << std::endl;
	std::cout << "parameters.spin_down_occup = " << parameters.spin_down_occup << std::endl;
}

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
		double result = 0.0;
		for (int r = 0; r < parameters.steps; r++) {
			double spectral = (parameters.j1 * (gf_local.at(r)(i, i) - std::conj(gf_local.at(r)(i, i)))).real();
			result += spectral;
 		}
		result = result * delta_energy / (2.0 * M_PI);
		std::cout << "For the atom number "<< i << " the spectral function integrates to " << result << std::endl;
	}
}




int main()
{
	Parameters parameters = Parameters::init();
	print_parameters(parameters);
	std::vector<double> kx(parameters.num_kx_points, 0);
	std::vector<double> ky(parameters.num_ky_points, 0);

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

	std::vector<dcomp> current_up(parameters.NIV_points, 0);
	std::vector<dcomp> current_down(parameters.NIV_points, 0);

	std::vector<dcomp> coherent_current_up(parameters.NIV_points, 0);
	std::vector<dcomp> coherent_current_down(parameters.NIV_points, 0);

	std::vector<dcomp> noncoherent_current_up(parameters.NIV_points, 0);
	std::vector<dcomp> noncoherent_current_down(parameters.NIV_points, 0);

	std::vector<dcomp> current_up_right(parameters.NIV_points, 0);
	std::vector<dcomp> current_up_left(parameters.NIV_points, 0);
	std::vector<dcomp> current_down_right(parameters.NIV_points, 0);
	std::vector<dcomp> current_down_left(parameters.NIV_points, 0);

	for (int m = 1; m < parameters.NIV_points; m++) {
		std::vector<std::vector<Eigen::MatrixXcd>> hamiltonian(
		    parameters.num_kx_points, std::vector<Eigen::MatrixXcd>(parameters.num_ky_points, Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length)));


		std::vector<Eigen::MatrixXcd> gf_local_up(parameters.steps, Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
		std::vector<Eigen::MatrixXcd> gf_local_down(parameters.steps, Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
		std::vector<Eigen::MatrixXcd> gf_local_lesser_up(parameters.steps, Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
		std::vector<Eigen::MatrixXcd> gf_local_lesser_down(parameters.steps, Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
		std::vector<std::vector<dcomp>> self_energy_mb_up(2 * parameters.chain_length, std::vector<dcomp>(parameters.steps)),
		    self_energy_mb_lesser_up(2 * parameters.chain_length, std::vector<dcomp>(parameters.steps, 0));
		std::vector<std::vector<dcomp>> self_energy_mb_down(2 * parameters.chain_length, std::vector<dcomp>(parameters.steps)),
		    self_energy_mb_lesser_down(2 * parameters.chain_length, std::vector<dcomp>(parameters.steps, 0));

		set_initial_spin(parameters, self_energy_mb_up, self_energy_mb_down);

		std::cout << "\n";
		std::cout << std::setprecision(15) << "The voltage difference is " << parameters.voltage_l[m] - parameters.voltage_r[m] << std::endl;


		std::vector<std::vector<EmbeddingSelfEnergy>> leads;
		for (int i = 0; i < parameters.num_kx_points; i++) {
			std::vector<EmbeddingSelfEnergy> vy;
			for (int j = 0; j < parameters.num_ky_points; j++) {
				vy.push_back(EmbeddingSelfEnergy(parameters, kx.at(i), ky.at(j), m));
			}
			leads.push_back(vy);
		}

		std::cout << "leads size: " << leads.at(0).size() << '\n';
		//get_k_averaged_embedding_self_energy(parameters, leads);

		for (int kx_i = 0; kx_i < parameters.num_kx_points; kx_i++) {
			for (int ky_i = 0; ky_i < parameters.num_ky_points; ky_i++) {
				get_hamiltonian(parameters, m, kx.at(kx_i), ky.at(ky_i), hamiltonian.at(kx_i).at(ky_i));
				//std::cout << "The hamiltonian is " <<  std::endl;
				//std::cout << hamiltonian.at(kx_i).at(ky_i) << std::endl;
				//std::cout << std::endl;
			}
		}
		//get_spectral_embedding_self_energy(parameters, leads, m);

		std::cout << "leads complete" << std::endl;
		get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, m, hamiltonian);
		get_local_gf_r_and_lesser(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, gf_local_down, gf_local_lesser_down, m, hamiltonian);
		std::cout << "got local retarded and lesser gf" << std::endl;

		/*
		std::ostringstream oss1gf;
		oss1gf << "textfiles/" << m << ".dos_non_int.txt";
		std::string var1 = oss1gf.str();
		std::ofstream dos_file_non_int;
		dos_file_non_int.open(var1);
		// myfile << parameters.steps << std::endl;
		for (int r = 0; r < parameters.steps; r++) {
			double dos_total_up = 0.0;
			double dos_total_down = 0.0;
			for (int i = 0; i < parameters.chain_length; i++) {
				dos_total_up += -gf_local_up.at(r)(i, i).imag();
				dos_total_down += -gf_local_down.at(r)(i, i).imag();
			}
			dos_file_non_int << parameters.energy.at(r) << "  " << dos_total_up << "   " << dos_total_down << " \n";
		}
		dos_file_non_int.close();
		*/


		std::vector<double> spins_occup(4 * parameters.chain_length); //the first 2 * chain_length is the spin up, the next 2 * chain_length is spin down.

		dmft(parameters, m, self_energy_mb_up, self_energy_mb_down, self_energy_mb_lesser_up, self_energy_mb_lesser_down, gf_local_up, gf_local_down, gf_local_lesser_up,
		    gf_local_lesser_down, leads, spins_occup, hamiltonian);

		std::cout << "got self energy " << std::endl;

		std::cout << "The difference between G-lesser and the fluctuation dissaption theorem is " << 
			get_gf_lesser_fd(parameters, gf_local_up, gf_local_lesser_up) << std::endl;

		if(parameters.hubbard_interaction == 0.0 && parameters.num_kx_points == 1 && parameters.num_ky_points == 1 && 
			parameters.num_ins_left == 0 && parameters.ins_metal_ins == true){
				analytic_gf(parameters, gf_local_up);
		}



		for (int i = 0; i < 2 * parameters.chain_length; i++) {
			if (parameters.atom_type.at(i) != 0) {
				std::cout << "The spin up occupation at atom " << i << " is " << spins_occup.at(i) << std::endl;
				std::cout << "The spin down occupation at atom " << i << " is " << spins_occup.at(i + 2 * parameters.chain_length) << std::endl;				
			}
		}

		std::vector<dcomp> transmission_up(parameters.steps, 0);
		std::vector<dcomp> transmission_down(parameters.steps, 0);
		if (parameters.hubbard_interaction == 0) {
			get_transmission(parameters, self_energy_mb_up, self_energy_mb_down, leads, transmission_up, transmission_down, m, hamiltonian);
			std::cout << "got transmission\n";
			get_landauer_buttiker_current(parameters, transmission_up, transmission_down, &current_up.at(m), &current_down.at(m), m);
			get_meir_wingreen_current(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m, &current_up_left.at(m), &current_up_right.at(m), hamiltonian);
			get_meir_wingreen_current(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, m, &current_down_left.at(m), &current_down_right.at(m), hamiltonian);
	
			std::cout << "The spin up current is " << current_up.at(m) << "\n" <<
					 "The spin down current is " << current_down.at(m) << "\n" << "\n";		
		} else {
			get_meir_wingreen_current(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m, &current_up_left.at(m), &current_up_right.at(m), hamiltonian);
			get_meir_wingreen_current(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, m, &current_down_left.at(m), &current_down_right.at(m), hamiltonian);
			get_transmission(parameters, self_energy_mb_up, self_energy_mb_down, leads, transmission_up, transmission_down, m, hamiltonian);
			get_landauer_buttiker_current(parameters, transmission_up, transmission_down, &coherent_current_up.at(m), &coherent_current_down.at(m), m);

			noncoherent_current_up.at(m) = 0.5 * (current_up_left.at(m) - current_up_right.at(m)) - coherent_current_up.at(m);
			noncoherent_current_down.at(m) = 0.5 * (current_down_left.at(m) - current_down_right.at(m)) - coherent_current_down.at(m);
		}


		std::cout << "The spin up left current is " << current_up_left.at(m) << "\n" <<
					 "The spin up right current is " << current_up_right.at(m) << "\n" <<
					 "The spin down left current is " << current_down_left.at(m) << "\n" <<
					 "The spin up right current is " << current_down_right.at(m) << "\n" <<
					 "The total current is " << 0.5 * (current_down_left.at(m) - current_down_right.at(m)) << "\n" <<
					 "The coherent current is " << coherent_current_down.at(m) << "\n" <<
					 "The noncoherent current is " << noncoherent_current_down.at(m) << "\n";

		std::cout << "\n";
		for (int i = 0; i < 2 * parameters.chain_length; i++) {
			std::ostringstream ossgf;
			ossgf << "textfiles/" << m << "." << i << ".gf.txt";
			std::string var = ossgf.str();

			std::ofstream gf_local_file;

			gf_local_file.open(var);

			for (int r = 0; r < parameters.steps; r++) {
				if (abs(gf_local_up.at(r)(i, parameters.num_ins_left).imag() - gf_local_up.at(r)(parameters.num_ins_left, i).imag()) > 0.00001
				    || abs(gf_local_up.at(r)(i, parameters.num_ins_left).real() - gf_local_up.at(r)(parameters.num_ins_left, i).real()) > 0.00001) {

					  std::cout << gf_local_up.at(r)(i, parameters.num_ins_left) << " " << gf_local_up.at(r)(parameters.num_ins_left, i) << " " << gf_local_up.at(r)(i, parameters.num_ins_left) - gf_local_up.at(r)(parameters.num_ins_left, i) << std::endl;
				}
        
				gf_local_file << parameters.energy.at(r) << "  " << gf_local_up.at(r)(i, i).real() << "   " << gf_local_up.at(r)(i, i).imag() << "   "
				              << gf_local_down.at(r)(i, i).real() << "   " << gf_local_down.at(r)(i, i).imag() << " \n";

				// std::cout << leads.self_energy_left.at(r) << "\n";
			}
			gf_local_file.close();
		}
		for (int i = 0; i < 2 * parameters.chain_length; i++) {
			std::ostringstream ossgf;
			ossgf << "textfiles/" << m << "." << i << ".gf_lesser.txt";
			std::string var = ossgf.str();

			std::ofstream gf_lesser_file;
			gf_lesser_file.open(var);
			for (int r = 0; r < parameters.steps; r++) {
				gf_lesser_file << parameters.energy.at(r) << "  " << gf_local_lesser_up.at(r)(i, i).real() << "   " << gf_local_lesser_up.at(r)(i, i).imag() << "   "
				               << gf_local_lesser_down.at(r)(i, i).real() << "   " << gf_local_lesser_down.at(r)(i, i).imag() << " "
				               << -2.0 * fermi_function(parameters.energy.at(r), parameters) * gf_local_down.at(r)(i, i).imag() << "\n";
			}
			gf_lesser_file.close();
		}

		integrate_spectral(parameters, gf_local_up);

		std::ostringstream oss;
		oss << "textfiles/" << m << ".tranmission.txt";
		std::string var = oss.str();

		std::ofstream transmission_file;
		transmission_file.open(var);
		// myfile << parameters.steps << std::endl;
		for (int r = 0; r < parameters.steps; r++) {
			transmission_file << parameters.energy.at(r) << "  " << transmission_up.at(r).real() << "  " << transmission_down.at(r).real()
			                  << "\n";
		}
		transmission_file.close();
		std::ostringstream oss1;
		oss1 << "textfiles/" << m << ".dos.txt";
		var = oss1.str();

		std::ofstream dos_file;
		dos_file.open(var);
		// myfile << parameters.steps << std::endl;
		for (int r = 0; r < parameters.steps; r++) {
			double dos_total_up = 0.0;
			double dos_total_down = 0.0;
			for (int i = 0; i < 2 * parameters.chain_length; i++) {
				dos_total_up += -gf_local_up.at(r)(i, i).imag();
				dos_total_down += -gf_local_down.at(r)(i, i).imag();
			}
			dos_file << parameters.energy.at(r) << "  " << dos_total_up << "   " << dos_total_down << " \n";
		}
		dos_file.close();

		for (int i = 0; i < 2 * parameters.chain_length; i++) {
			std::ostringstream ossser;
			if (parameters.atom_type.at(i) == 0){
				continue;
			}
			ossser << "textfiles/" << m << "." << i << ".se_r.txt";
			var = ossser.str();
			std::ofstream se_r_file;
			se_r_file.open(var);
			for (int r = 0; r < parameters.steps; r++) {
				se_r_file << parameters.energy.at(r) << "  " << self_energy_mb_up.at(i).at(r).real() << "  " << self_energy_mb_up.at(i).at(r).imag() << "  "
				          << self_energy_mb_down.at(i).at(r).real() << "  " << self_energy_mb_down.at(i).at(r).imag() << "\n";
			}
			se_r_file.close();
		}

		for (int i = 0; i < 2 * parameters.chain_length; i++) {
			std::ostringstream ossser;
			if (parameters.atom_type.at(i) == 0){
				continue;
			}
			ossser << "textfiles/" << m << "." << i << ".se_l.txt";
			var = ossser.str();
			std::ofstream se_lesser_file;
			se_lesser_file.open(var);
			for (int r = 0; r < parameters.steps; r++) {
				dcomp x = -2.0 * fermi_function(parameters.energy.at(r), parameters) * (self_energy_mb_up.at(i).at(r).imag());
				se_lesser_file << parameters.energy.at(r) << "  " << self_energy_mb_lesser_up.at(i).at(r).real() << " " << self_energy_mb_lesser_up.at(i).at(r).imag() << " "
				               << x.imag() << "\n";
			}
			se_lesser_file.close();
		}
	}
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
			             << 2 * current_up_left.at(m).real() << "   " << 2 * current_up_right.at(m).real()
			             << "   "
			             //<< current_down_left.at(m).real() << "   "
			             << 2 * current_up_left.at(m).real() + 2 * current_up_right.at(m).real() << "\n";
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

			current_file << parameters.voltage_l[m] - parameters.voltage_r[m] << "   " << current_up_left.at(m).real() << "   " << current_up_right.at(m).real() << "   "
			             << current_down_left.at(m).real() << "   " << current_down_right.at(m).real() << "     " << 0.5 * (current_down_left.at(m) - current_down_right.at(m))
			             << "  " << coherent_current_down.at(m) << "   " << noncoherent_current_down.at(m) << "\n";
		}
		current_file.close();
	}

	return 0;
}