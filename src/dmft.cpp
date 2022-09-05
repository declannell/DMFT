#include "dmft.h"

#include <mpi.h>

#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <vector>

#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"

void get_spin_occupation(const Parameters &parameters, const std::vector<dcomp> &gf_lesser_up,
                        const std::vector<dcomp> &gf_lesser_down, double *spin_up, double *spin_down)
{
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)(parameters.steps);
	double result_up = 0.0, result_down = 0.0;
	//for(int r = 0; r < parameters.steps; r++){
	//    result_up = (delta_energy) * gf_lesser_up.at(r).imag() + result_up;
	//    result_down = (delta_energy) * gf_lesser_down.at(r).imag() + result_down;
	//}

	for (int r = 0; r < parameters.steps; r++) {
		//std::cout << parameters.energy.at(r) << " " << gf_lesser_down.at(r).imag() <<  std::endl;
		if (r == 0 || r == parameters.steps - 1) {
			result_up += (delta_energy / 2.0) * gf_lesser_up.at(r).imag();
			result_down += (delta_energy / 2.0) * gf_lesser_down.at(r).imag();
		} else {
			result_up += (delta_energy) * gf_lesser_up.at(r).imag();
			result_down += (delta_energy) * gf_lesser_down.at(r).imag();
		}
	}

		*spin_up = 1.0 / (2.0 * M_PI) * result_up;
		*spin_down = 1.0 / (2.0 * M_PI) * result_down;
}
	//std::cout << *spin_up << std::endl;


void get_difference(const Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &old_green_function,
                double &difference, int &index){
	difference = -std::numeric_limits<double>::infinity();
	double old_difference = 0;
	double real_difference, imag_difference;
	for (int r = 0; r < parameters.steps; r++) {
		for (int i = 0; i < parameters.chain_length; i++) {
			for (int j = 0; j < parameters.chain_length; j++) {
				real_difference = abs(gf_local_up.at(r)(i, j).real() - old_green_function.at(r)(i, j).real());
				imag_difference = abs(gf_local_up.at(r)(i, j).imag() - old_green_function.at(r)(i, j).imag());
				//std::cout << real_difference << "  " << imag_difference << "  "  << difference << "\n";
				difference = std::max(difference, std::max(real_difference, imag_difference));
				old_green_function.at(r)(i, j) = gf_local_up.at(r)(i, j);
				if (difference > old_difference) {
					index = r;
				}
				old_difference = difference;
			}
		}
		//std::cout <<"\n";
	}
}

void fluctuation_dissipation(const Parameters &parameters, const std::vector<dcomp> &green_function, std::vector<dcomp> &lesser_green_function)
{
	for (int r = 0; r < parameters.steps; r++) {
		lesser_green_function.at(r) = -1.0 * fermi_function(parameters.energy.at(r), parameters) * (green_function.at(r) - std::conj(green_function.at(r)));
		//std::cout << lesser_green_function.at(r) << std::endl;
	}
}

dcomp integrate(const Parameters& parameters, const std::vector<dcomp>& gf_1, const std::vector<dcomp>& gf_2, const std::vector<dcomp>& gf_3, const int r)
{
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
	dcomp result = 0;
	for (int i = 0; i < parameters.steps; i++) {
		for (int j = 0; j < parameters.steps; j++) {
			if (((i + j - r) > 0) && ((i + j - r) < parameters.steps)) {
				//this integrates the equation in PHYSICAL REVIEW B 74, 155125 2006
				//I say the green function is zero outside e_lower_bound and e_upper_bound. This means I need the final green function in the integral to be within an energy of e_lower_bound
				//and e_upper_bound. The index of 0 corresponds to e_lower_bound. Hence we need i+J-r>0 but in order to be less an energy of e_upper_bound we need i+j-r<steps.
				//These conditions ensure the enrgy of the gf3 greens function to be within (e_upper_bound, e_lower_bound)
				result += (delta_energy / (2.0 * M_PI)) * (delta_energy / (2.0 * M_PI)) * gf_1.at(i) * gf_2.at(j) * gf_3.at(i + j - r);
			}
		}
	}
	return result;
}

double kramer_kronig_relation(const Parameters& parameters, std::vector<double>& impurity_self_energy_imag, int r)
{
	double real_self_energy = 0;
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
	for (int q = 0; q < parameters.steps; q++) {
		if (q != r) {
			real_self_energy += delta_energy * impurity_self_energy_imag.at(q) / (parameters.energy.at(q) - parameters.energy.at(r));
        }
    }
	return real_self_energy / M_PI;
}

double get_prefactor(const int i, const int j, const int r, const int voltage_step, const Parameters &parameters, const std::vector<std::vector<dcomp>>   &self_energy_mb_up,
    const std::vector<std::vector<dcomp>>   &self_energy_mb_down, const std::vector<std::vector<dcomp>>  &self_energy_mb_lesser_up,
    const std::vector<std::vector<dcomp>>   &self_energy_mb_lesser_down, const std::vector<std::vector<EmbeddingSelfEnergy>> &leads){
	
	double prefactor;
	if (voltage_step == 0){
		prefactor = fermi_function(parameters.energy.at(i), parameters) * fermi_function(parameters.energy.at(j), parameters) 
			+ (1 - fermi_function(parameters.energy.at(i), parameters) - fermi_function(parameters.energy.at(j), parameters)) *
			 fermi_function(parameters.energy.at(j) + parameters.energy.at(i) - parameters.energy.at(r), parameters); 
	} else {
		int a = i + j - r;
		double numerator = - 2.0 * fermi_function(parameters.energy.at(i) - parameters.voltage_l.at(voltage_step), parameters) * 
                            leads.at(0).at(0).self_energy_left.at(i).imag()  - 2.0 * fermi_function(parameters.energy.at(i) - parameters.voltage_r.at(voltage_step), parameters) * 
                            leads.at(0).at(0).self_energy_right.at(i).imag();
		double deminator = - leads.at(0).at(0).self_energy_left.at(i).imag()  - leads.at(0).at(0).self_energy_right.at(i).imag();
		double prefactor_1 = 0.5 * (numerator + self_energy_mb_lesser_up.at(parameters.num_ins_left).at(i).imag()) / (deminator - self_energy_mb_up.at(parameters.num_ins_left).at(i).imag());

		numerator = - 2.0 * fermi_function(parameters.energy.at(j) - parameters.voltage_l.at(voltage_step), parameters) * 
                            leads.at(0).at(0).self_energy_left.at(j).imag()  - 2.0 * fermi_function(parameters.energy.at(j) - parameters.voltage_r.at(voltage_step), parameters) * 
                            leads.at(0).at(0).self_energy_right.at(j).imag();
		deminator = - leads.at(0).at(0).self_energy_left.at(j).imag()  - leads.at(0).at(0).self_energy_right.at(j).imag();
		double prefactor_2 = 0.5 * (numerator + self_energy_mb_lesser_down.at(parameters.num_ins_left).at(j).imag()) / (deminator - self_energy_mb_down.at(parameters.num_ins_left).at(j).imag());

		numerator = - 2.0 * fermi_function(parameters.energy.at(a) - parameters.voltage_l.at(voltage_step), parameters) * 
                            leads.at(0).at(0).self_energy_left.at(a).imag()  - 2.0 * fermi_function(parameters.energy.at(a) - parameters.voltage_r.at(voltage_step), parameters) * 
                            leads.at(0).at(0).self_energy_right.at(a).imag();
		deminator = - leads.at(0).at(0).self_energy_left.at(a).imag()  - leads.at(0).at(0).self_energy_right.at(a).imag();
		double prefactor_3 = 0.5 * (numerator + self_energy_mb_lesser_down.at(parameters.num_ins_left).at(a).imag()) / (deminator - self_energy_mb_down.at(parameters.num_ins_left).at(a).imag());
		prefactor = prefactor_1 * prefactor_2 + (1.0 - prefactor_1 - prefactor_2) * prefactor_3;
	}		
	return prefactor;
}

double integrate_equilibrium(const Parameters& parameters, const std::vector<double>& gf_1, const std::vector<double>& gf_2, const std::vector<double>& gf_3, const int r,
    const std::vector<std::vector<dcomp>>  &self_energy_mb_up, const std::vector<std::vector<dcomp>>  &self_energy_mb_down,
    const std::vector<std::vector<dcomp>>   &self_energy_mb_lesser_up, const std::vector<std::vector<dcomp>>   &self_energy_mb_lesser_down, int voltage_step,
	const std::vector<std::vector<EmbeddingSelfEnergy>> &leads)
{
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
	double result = 0;
	for (int i = 0; i < parameters.steps; i++) {
		for (int j = 0; j < parameters.steps; j++) {
			if (((i + j - r) > 0) && ((i + j - r) < parameters.steps)) {
				double prefactor = get_prefactor(i, j, r, voltage_step, parameters, self_energy_mb_up, self_energy_mb_down,
    				self_energy_mb_lesser_up, self_energy_mb_lesser_down, leads);
				//this integrates the equation in PHYSICAL REVIEW B 74, 155125 2006
				//I say the green function is zero outside e_lower_bound and e_upper_bound. This means I need the final green function in the integral to be within an energy of e_lower_bound
				//and e_upper_bound. The index of 0 corresponds to e_lower_bound. Hence we need i+J-r>0 but in order to be less an energy of e_upper_bound we need i+j-r<steps.
				//These conditions ensure the enrgy of the gf3 greens function to be within (e_upper_bound, e_lower_bound)
				result += prefactor * (delta_energy / (M_PI)) * (delta_energy / (M_PI)) * gf_1.at(i) * gf_2.at(j) * gf_3.at(i + j - r);
			}
		}
	}
	return result;
}

void self_energy_2nd_order_kramers_kronig(const Parameters& parameters, const std::vector<dcomp>& impurity_gf_up, const std::vector<dcomp>& impurity_gf_down,
    const std::vector<dcomp>& impurity_gf_up_lesser, const std::vector<dcomp>& impurity_gf_down_lesser, std::vector<dcomp>& impurity_self_energy,
    std::vector<dcomp>& impurity_self_energy_lesser_up, const int voltage_step, const std::vector<std::vector<dcomp>>   &self_energy_mb_up, const std::vector<std::vector<dcomp>>   &self_energy_mb_down,
    const std::vector<std::vector<dcomp>>   &self_energy_mb_lesser_up, const std::vector<std::vector<dcomp>>   &self_energy_mb_lesser_down,
	const std::vector<std::vector<EmbeddingSelfEnergy>> &leads)
{
	std::vector<double> impurity_self_energy_real(parameters.steps), impurity_self_energy_imag(parameters.steps);

	std::vector<dcomp> impurity_gf_down_advanced_imag(parameters.steps), gf_greater_down(parameters.steps);
	std::vector<double> impurity_gf_up_imag(parameters.steps), impurity_gf_down_imag(parameters.steps);
	//std::vector<double> impurity_gf_up_real(parameters.steps), impurity_gf_down_real(parameters.steps);
	for (int r = 0; r < parameters.steps; r++) {
		//impurity_gf_down_advanced_imag.at(r) = parameters.j1 * std::conj(impurity_gf_down.at(r)).imag();
		gf_greater_down.at(r) = impurity_gf_down_lesser.at(r) + (impurity_gf_down.at(r) - std::conj(impurity_gf_down.at(r)));
		impurity_gf_up_imag.at(r) = impurity_gf_up.at(r).imag();
		impurity_gf_down_imag.at(r) = impurity_gf_down.at(r).imag();
		//impurity_gf_up_real.at(r) = impurity_gf_up.at(r).real();
		//impurity_gf_down_real.at(r) = impurity_gf_down.at(r).real();
    }
    //I only want to calculate the imaginary part of the self energy.

	for (int r = 0; r < parameters.steps; r++){
		impurity_self_energy_imag.at(r) = parameters.hubbard_interaction * parameters.hubbard_interaction
		    * integrate_equilibrium(parameters, impurity_gf_up_imag, impurity_gf_down_imag, impurity_gf_down_imag, r, self_energy_mb_up, self_energy_mb_down,
    				self_energy_mb_lesser_up, self_energy_mb_lesser_down, voltage_step, leads); 	

		impurity_self_energy_lesser_up.at(r) =
		    parameters.hubbard_interaction * parameters.hubbard_interaction * (integrate(parameters, impurity_gf_up_lesser, impurity_gf_down_lesser, gf_greater_down, r));
	}
	std::ostringstream ossser;
	ossser << "textfiles/"
	       << "se_krammer_kronig.txt";
	std::string var = ossser.str();
	std::ofstream se_krammer_kronig;
	se_krammer_kronig.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		impurity_self_energy_real.at(r) = kramer_kronig_relation(parameters, impurity_self_energy_imag, r);
		impurity_self_energy.at(r) = impurity_self_energy_real.at(r) + parameters.j1 * impurity_self_energy_imag.at(r);

		se_krammer_kronig << parameters.energy.at(r) << "  " << impurity_self_energy_real.at(r) << "  " << impurity_self_energy_imag.at(r) << "  " << impurity_self_energy.at(r)
		                  << "\n";
	}
	se_krammer_kronig.close();
}

void impurity_solver(const Parameters &parameters, const int voltage_step, const std::vector<dcomp>  &impurity_gf_up, const std::vector<dcomp>  &impurity_gf_down,
    const std::vector<dcomp>  &impurity_gf_lesser_up, const std::vector<dcomp>  &impurity_gf_lesser_down,
    std::vector<dcomp>  &impurity_self_energy_up, std::vector<dcomp>  &impurity_self_energy_down, 
    std::vector<dcomp>  &impurity_self_energy_lesser_up, std::vector<dcomp>  &impurity_self_energy_lesser_down,
    double *spin_up, double *spin_down, const std::vector<std::vector<dcomp>> &self_energy_mb_up, const std::vector<std::vector<dcomp>>  &self_energy_mb_down,
    const std::vector<std::vector<dcomp>>  &self_energy_mb_lesser_up, const std::vector<std::vector<dcomp>>  &self_energy_mb_lesser_down,
	const std::vector<std::vector<EmbeddingSelfEnergy>> &leads)
{
	get_spin_occupation(parameters, impurity_gf_lesser_up, impurity_gf_lesser_down, spin_up, spin_down);
	std::cout << std::setprecision(15) << "The spin up occupancy is " << *spin_up << "\n";
	std::cout << "The spin down occupancy is " << *spin_down << "\n";

	if (parameters.interaction_order == 2) {

        //one can choose a normal intergation (self_energy_2nd_order) or a krammer kronig method (self_energy_2nd_order_krammer_kronig)
		self_energy_2nd_order_kramers_kronig(
		    parameters, impurity_gf_up, impurity_gf_down, impurity_gf_lesser_up, impurity_gf_lesser_down, impurity_self_energy_up, impurity_self_energy_lesser_up, voltage_step, 
				self_energy_mb_up, self_energy_mb_down, self_energy_mb_lesser_up, self_energy_mb_lesser_down, leads);
		self_energy_2nd_order_kramers_kronig(
		    parameters, impurity_gf_down, impurity_gf_up, impurity_gf_lesser_down, impurity_gf_lesser_up, impurity_self_energy_down, impurity_self_energy_lesser_down, voltage_step,
				self_energy_mb_up, self_energy_mb_down, self_energy_mb_lesser_up, self_energy_mb_lesser_down, leads);

		for (int r = 0; r < parameters.steps; r++) {
			impurity_self_energy_up.at(r) += parameters.hubbard_interaction * (*spin_down);
			impurity_self_energy_down.at(r) += parameters.hubbard_interaction * (*spin_up);
		}
	}

	if (parameters.interaction_order == 1) {
		for (int r = 0; r < parameters.steps; r++) {
			impurity_self_energy_up.at(r) = parameters.hubbard_interaction * (*spin_down);
			impurity_self_energy_down.at(r) = parameters.hubbard_interaction * (*spin_up);
		}
	}
}

void dmft(const Parameters &parameters, const int voltage_step, 
        std::vector<std::vector<dcomp>> &self_energy_mb_up, std::vector<std::vector<dcomp>> &self_energy_mb_down, 
        std::vector<std::vector<dcomp>> &self_energy_mb_lesser_up, std::vector<std::vector<dcomp>> &self_energy_mb_lesser_down,
        std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &gf_local_down,
        std::vector<Eigen::MatrixXcd> &gf_local_lesser_up, std::vector<Eigen::MatrixXcd> &gf_local_lesser_down,
        const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<double> &spins_occup, const std::vector<std::vector<Eigen::MatrixXd>> &hamiltonian)
{
	double difference = std::numeric_limits<double>::infinity();
	int index, count = 0;

	std::vector<Eigen::MatrixXcd> old_green_function(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
	while (difference > parameters.convergence && count < parameters.self_consistent_steps) {
		get_difference(parameters, gf_local_up, old_green_function, difference, index);
		std::cout << "The difference is " << difference << ". The count is " << count << std::endl;
		if (difference < parameters.convergence) {
			break;
		}

		for (int j = 0; j < parameters.num_cor; j++) {  //we only do the dmft loop over the correlated metal.
			int i = parameters.num_ins_left + j;
			std::vector<dcomp> diag_gf_local_up(parameters.steps), diag_gf_local_down(parameters.steps), diag_gf_local_lesser_up(parameters.steps),
			    diag_gf_local_lesser_down(parameters.steps), impurity_self_energy_up(parameters.steps), impurity_self_energy_down(parameters.steps),
			    impurity_self_energy_lesser_up(parameters.steps), impurity_self_energy_lesser_down(parameters.steps);
			std::cout << "atom which we put on correlation" << i << std::endl;
			for (int r = 0; r < parameters.steps; r++) {
				diag_gf_local_up.at(r) = gf_local_up.at(r)(i, i);
				diag_gf_local_down.at(r) = gf_local_down.at(r)(i, i);
				diag_gf_local_lesser_up.at(r) = gf_local_lesser_up.at(r)(i, i);
				diag_gf_local_lesser_down.at(r) = gf_local_lesser_down.at(r)(i, i);
			}

			impurity_solver(parameters, voltage_step, diag_gf_local_up, diag_gf_local_down, diag_gf_local_lesser_up, diag_gf_local_lesser_down, impurity_self_energy_up,
			    impurity_self_energy_down, impurity_self_energy_lesser_up, impurity_self_energy_lesser_down, &spins_occup.at(i), &spins_occup.at(i + parameters.chain_length),
				self_energy_mb_up, self_energy_mb_down, self_energy_mb_lesser_up, self_energy_mb_lesser_down, leads);

            if(count == 0){
                for (int r = 0; r < parameters.steps; r++) {
                    self_energy_mb_up.at(i).at(r) = impurity_self_energy_up.at(r);
                    self_energy_mb_down.at(i).at(r) = impurity_self_energy_down.at(r);
                    self_energy_mb_lesser_up.at(i).at(r) = impurity_self_energy_lesser_up.at(r);
                    self_energy_mb_lesser_down.at(i).at(r) = impurity_self_energy_lesser_down.at(r);
                }
            } else {
                for (int r = 0; r < parameters.steps; r++) {
                    self_energy_mb_up.at(i).at(r) = (impurity_self_energy_up.at(r) + self_energy_mb_up.at(i).at(r)) * 0.5;
                    self_energy_mb_down.at(i).at(r) = (impurity_self_energy_down.at(r) + self_energy_mb_down.at(i).at(r)) * 0.5;
                    self_energy_mb_lesser_up.at(i).at(r) = (impurity_self_energy_lesser_up.at(r) + self_energy_mb_lesser_up.at(i).at(r)) * 0.5;
                    self_energy_mb_lesser_down.at(i).at(r) = (impurity_self_energy_lesser_down.at(r) + self_energy_mb_lesser_down.at(i).at(r)) * 0.5;
                }
            }
		
		}
		get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, voltage_step, hamiltonian);
		get_local_gf_r_and_lesser(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, gf_local_down, gf_local_lesser_down, voltage_step, hamiltonian);

		//for (int i = 0; i < parameters.chain_length; i++)  {
		//    std::ostringstream ossgf;
		//    ossgf << "textfiles/" << count << "." << i << ".gf.txt";
		//    std::string var = ossgf.str();
		//
		//    std::ofstream gf_local_file;
		//    gf_local_file.open(var);
		//    for (int r = 0; r < parameters.steps; r++) {
		//      gf_local_file << parameters.energy.at(r) << "  "
		//                    << gf_local_up.at(r)(i, i).real() << "   "
		//                    << gf_local_up.at(r)(i, i).imag() << "   "
		//                    << gf_local_down.at(r)(i, i).real() << "   "
		//                    << gf_local_down.at(r)(i, i).imag() << " \n";
		//
		//      // std::cout << leads.self_energy_left.at(r) << "\n";
		//    }
		//    gf_local_file.close();
		//}
		////
		//if(voltage_step == 0){
		//    std::vector<Eigen::MatrixXcd> gf_local_lesser_up_FD(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
		//    for(int r = 0; r < parameters.steps; r++){
		//        for(int i = 0; i < parameters.chain_length; i++){
		//            for(int j = 0; j < parameters.chain_length; j++){
		//                gf_local_lesser_up_FD.at(r)(i, j) = - 1.0 * fermi_function(parameters.energy.at(r), parameters) *
		//                    (gf_local_up.at(r)(i, j) - std::conj(gf_local_up.at(r)(j, i)));
		//
		//            }
		//        }
		//        //std::cout << gf_local_lesser_up_FD.at(r) << std::endl;
		//    }
		//    double difference_fd = 0;
		//
		//    get_difference(parameters, gf_local_lesser_up, gf_local_lesser_up_FD, difference_fd, index);
		//    //i need to do this again as the difference function will overwrite gf_local_lesser_up_FD
		//    for(int r = 0; r < parameters.steps; r++){
		//        for(int i = 0; i < parameters.chain_length; i++){
		//            for(int j = 0; j < parameters.chain_length; j++){
		//                gf_local_lesser_up_FD.at(r)(i, j) = - 1.0 * fermi_function(parameters.energy.at(r), parameters) *
		//                    (gf_local_up.at(r)(i, j) - std::conj(gf_local_up.at(r)(j, i)));
		//
		//            }
		//        }
		//        //std::cout << gf_local_lesser_up_FD.at(r) << std::endl;
		//    }
		//
		//    std::cout << "The difference between the fD and other is " << difference_fd << std::endl;
		//    std::cout << "The index is " << index << std::endl;
		//    std::cout << gf_local_lesser_up.at(index) << gf_local_lesser_up_FD.at(index) <<  gf_local_lesser_up.at(index) - gf_local_lesser_up_FD.at(index)  << std::endl;
		//
		//
		//    std::ofstream gf_lesser_file;
		//    gf_lesser_file.open("textfiles/gf_lesser_c++.txt");
		//    // myfile << parameters.steps << std::endl;
		//    difference_fd = 0;
		//    for(int i = 0; i < parameters.chain_length; i++){
		//        for (int r = 0; r < parameters.steps; r++)
		//        {
		//            gf_lesser_file << parameters.energy.at(r) << "  " << gf_local_lesser_up.at(r)(i, i).real() << "  " << gf_local_lesser_up.at(r)(i, i).imag()
		//                << "  " << gf_local_lesser_up_FD.at(r)(i, i).real() << "  " << gf_local_lesser_up_FD.at(r)(i, i).imag() << "  "
		//                <<   gf_local_lesser_up.at(r)(i, i).imag() - gf_local_lesser_up_FD.at(r)(i, i).imag() << "\n";
		//
		//            difference_fd += abs(gf_local_lesser_up.at(r)(i, i).imag() - gf_local_lesser_up_FD.at(r)(i, i).imag());
		//        }
		//    }
		//    gf_lesser_file.close();
		//    std::cout << "The total difference between the two methods is " << difference_fd << std::endl;
		//}
		//
		count++;
	}
}
