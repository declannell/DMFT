#include "dmft.h"

#include <mpi.h>

#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <vector>

#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"

void get_spin_occupation(Parameters& parameters, std::vector<dcomp>& gf_lesser_up, std::vector<dcomp>& gf_lesser_down, double* spin_up, double* spin_down)
{
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
	double result_up = 0.0, result_down = 0.0;
	//for(int r = 0; r < parameters.steps; r++){
	//    result_up = (delta_energy) * gf_lesser_up.at(r).imag() + result_up;
	//    result_down = (delta_energy) * gf_lesser_down.at(r).imag() + result_down;
	//}

	for (int r = 0; r < parameters.steps; r++) {
		if (r == 0 || r == parameters.steps - 1) {
			result_up = (delta_energy / 2.0) * gf_lesser_up.at(r).imag() + result_up;
			result_down = (delta_energy / 2.0) * gf_lesser_down.at(r).imag() + result_down;
		} else {
			result_up = (delta_energy)*gf_lesser_up.at(r).imag() + result_up;
			result_down = (delta_energy)*gf_lesser_down.at(r).imag() + result_down;
		}
	}
	*spin_up = 1.0 / (2.0 * M_PI) * result_up;
	*spin_down = 1.0 / (2.0 * M_PI) * result_down;

	//std::cout << *spin_up << std::endl;
}

void get_difference(Parameters& parameters, std::vector<Eigen::MatrixXcd>& gf_local_up, std::vector<Eigen::MatrixXcd>& old_green_function, double& difference, int& index)
{
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

void fluctuation_dissipation(Parameters& parameters, const std::vector<dcomp>& green_function, std::vector<dcomp>& lesser_green_function)
{
	for (int r = 0; r < parameters.steps; r++) {
		lesser_green_function.at(r) = -1.0 * fermi_function(parameters.energy.at(r), parameters) * (green_function.at(r) - std::conj(green_function.at(r)));
		//std::cout << lesser_green_function.at(r) << std::endl;
	}
}

dcomp integrate(Parameters& parameters, std::vector<dcomp>& gf_1, std::vector<dcomp>& gf_2, std::vector<dcomp>& gf_3, int r)
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
				//
				//if ((i ==0 || i ==parameters.steps - 1) || ((j ==0 || j ==parameters.steps - 1))){
				//result = 0.5 * (delta_energy / (2.0 * M_PI)) * (delta_energy / (2.0 * M_PI)) *
				//    gf_1.at(i) * gf_2.at(j) * gf_3.at(i + j - r) + result;
				//} else if ((i ==0 || i ==parameters.steps - 1) && ((j ==0 || j ==parameters.steps - 1))){
				//result = 0.25 * (delta_energy / (2.0 * M_PI)) * (delta_energy / (2.0 * M_PI)) *
				//    gf_1.at(i) * gf_2.at(j) * gf_3.at(i + j - r) + result;
				//} else {
				//result = (delta_energy / (2.0 * M_PI)) * (delta_energy / (2.0 * M_PI)) *
				//    gf_1.at(i) * gf_2.at(j) * gf_3.at(i + j - r) + result;
				//}
			}
		}
	}
	return result;
}

void check_self_energy_fd(Parameters& parameters, std::vector<dcomp>& impurity_gf_up, std::vector<dcomp>& impurity_gf_down, std::vector<dcomp>& impurity_gf_up_lesser,
    std::vector<dcomp>& impurity_gf_down_lesser, std::vector<dcomp>& gf_greater_down)
{
	std::cout << "this check is happening" << std::endl;
	for (int r = 0; r < parameters.steps; r++) {
		std::cout << gf_greater_down.at(r) << std::endl;
	}
	for (int r = 0; r < parameters.steps; r++) {
		for (int i = 0; i < parameters.steps; i++) {
			for (int j = 0; j < parameters.steps; j++) {
				if (parameters.energy.at(i) + parameters.energy.at(j) - parameters.energy.at(r) >= parameters.e_lower_bound
				    && parameters.energy.at(i) + parameters.energy.at(j) - parameters.energy.at(r) < parameters.e_upper_bound - 0.1) {
					dcomp lesser = impurity_gf_up_lesser.at(i) * impurity_gf_down_lesser.at(j) * gf_greater_down.at(i + j - r);
					dcomp retarded = impurity_gf_up.at(i) * impurity_gf_down.at(j) * impurity_gf_down_lesser.at(i + j - r)
					    + impurity_gf_up.at(i) * impurity_gf_down_lesser.at(j) * impurity_gf_down_lesser.at(i + j - r)
					    + impurity_gf_up_lesser.at(i) * impurity_gf_down.at(j) * impurity_gf_down_lesser.at(i + j - r)
					    + impurity_gf_up_lesser.at(i) * impurity_gf_down_lesser.at(j) * std::conj(impurity_gf_down.at(i + j - r));

					//std::cout << lesser << "  " <<  "  " << i+j-r << std::endl;

					if (abs(lesser.imag() + 2.0 * fermi_function(parameters.energy.at(r), parameters) * retarded.imag()) > 0.000001) {
						std::cout << lesser << " " << retarded << " " << r << "  " << i << "  " << j << std::endl;
					}
				}
			}
		}
	}
}


double kramer_kronig_relation(Parameters& parameters, std::vector<double>& impurity_self_energy_imag, int r)
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

double integrate_equilibrium(Parameters& parameters, std::vector<double>& gf_1, std::vector<double>& gf_2, std::vector<double>& gf_3, int r)
{
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
	double result = 0;
	for (int i = 0; i < parameters.steps; i++) {
		for (int j = 0; j < parameters.steps; j++) {
			if (((i + j - r) > 0) && ((i + j - r) < parameters.steps)) {
				double prefactor = fermi_function(parameters.energy.at(i), parameters) * fermi_function(parameters.energy.at(j), parameters) 
					+ (1 - fermi_function(parameters.energy.at(i), parameters) - fermi_function(parameters.energy.at(j), parameters)) 
					* fermi_function(parameters.energy.at(j) + parameters.energy.at(i) - parameters.energy.at(r), parameters); 
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

void self_energy_2nd_order_kramers_kronig(Parameters& parameters, std::vector<dcomp>& impurity_gf_up, std::vector<dcomp>& impurity_gf_down,
    std::vector<dcomp>& impurity_gf_up_lesser, std::vector<dcomp>& impurity_gf_down_lesser, std::vector<dcomp>& impurity_self_energy,
    std::vector<dcomp>& impurity_self_energy_lesser_up)
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
		    * integrate_equilibrium(parameters, impurity_gf_up_imag, impurity_gf_down_imag, impurity_gf_down_imag, r); 		
		/*
		impurity_self_energy_imag.at(r) = parameters.hubbard_interaction * parameters.hubbard_interaction
		    * (integrate(parameters, impurity_gf_up_real, impurity_gf_down_real, impurity_gf_down_lesser, r).imag());  // line 
		impurity_self_energy_imag.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction
		    * (integrate(parameters, impurity_gf_up_imag, impurity_gf_down_imag, impurity_gf_down_lesser, r).imag());  // line 3

		impurity_self_energy_imag.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction
		    * (integrate(parameters, impurity_gf_up_imag, impurity_gf_down_lesser, impurity_gf_down_lesser, r).imag());  // line 2

		impurity_self_energy_imag.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction
		    * (integrate(parameters, impurity_gf_up_lesser, impurity_gf_down_imag, impurity_gf_down_lesser, r).imag());  // line 1

		impurity_self_energy_imag.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction
		    * (integrate(parameters, impurity_gf_up_lesser, impurity_gf_down_lesser, impurity_gf_down_advanced_imag, r).imag());  //line 4
		//there could be a problem here with the constant multiplying sigma lesser
	*/
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

void self_energy_2nd_order(Parameters& parameters, std::vector<dcomp>& impurity_gf_up, std::vector<dcomp>& impurity_gf_down, std::vector<dcomp>& impurity_gf_up_lesser,
    std::vector<dcomp>& impurity_gf_down_lesser, std::vector<dcomp>& impurity_self_energy, std::vector<dcomp>& impurity_self_energy_lesser_up)
{
	std::vector<dcomp> impurity_gf_down_advanced(parameters.steps), gf_greater_down(parameters.steps);

	std::ostringstream ossser;
	ossser << "textfiles/"
	       << "gf_greater.txt";
	std::string var = ossser.str();
	std::ofstream gf_greater_file;
	gf_greater_file.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		gf_greater_file << parameters.energy.at(r) << "  " << impurity_gf_down.at(r) << "  " << std::conj(impurity_gf_down.at(r)) << "  " << impurity_gf_down_lesser.at(r) << "  "
		                << parameters.j1 * 2.0 * impurity_gf_down.at(r).imag() + impurity_gf_down_lesser.at(r) << "  "
		                << (1 - fermi_function(parameters.energy.at(r), parameters)) * (impurity_gf_down.at(r) - std::conj(impurity_gf_down.at(r))) << "\n";

		impurity_gf_down_advanced.at(r) = std::conj(impurity_gf_down.at(r));
		gf_greater_down.at(r) = parameters.j1 * 2.0 * impurity_gf_down.at(r).imag() + impurity_gf_down_lesser.at(r);
	}
	gf_greater_file.close();

	for (int r = 0; r < parameters.steps; r++) {
		impurity_self_energy.at(r) =
		    parameters.hubbard_interaction * parameters.hubbard_interaction * (integrate(parameters, impurity_gf_up, impurity_gf_down, impurity_gf_down_lesser, r));  // line 3

		impurity_self_energy.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction
		    * (integrate(parameters, impurity_gf_up, impurity_gf_down_lesser, impurity_gf_down_lesser, r));  // line 2

		impurity_self_energy.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction
		    * (integrate(parameters, impurity_gf_up_lesser, impurity_gf_down, impurity_gf_down_lesser, r));  // line 1

		impurity_self_energy.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction
		    * (integrate(parameters, impurity_gf_up_lesser, impurity_gf_down_lesser, impurity_gf_down_advanced, r));  //line 4
		//there could be a problem here with the constant multiplying sigma lesser
		impurity_self_energy_lesser_up.at(r) = parameters.hubbard_interaction * parameters.hubbard_interaction
		    * (integrate(parameters, impurity_gf_up_lesser, impurity_gf_down_lesser, gf_greater_down, r));  //-2.0 * parameters.j1 *
		//fermi_function(parameters.energy.at(r), parameters) * impurity_self_energy.at(r).imag();//parameters.hubbard_interaction * parameters.hubbard_interaction *
		//(integrate(parameters, impurity_gf_up_lesser, impurity_gf_down_lesser, gf_greater_down, r));
	}
}




void impurity_solver(Parameters& parameters, int voltage_step, std::vector<dcomp>& impurity_gf_up, std::vector<dcomp>& impurity_gf_down, std::vector<dcomp>& impurity_gf_up_lesser,
    std::vector<dcomp>& impurity_gf_down_lesser, std::vector<dcomp>& impurity_self_energy_up, std::vector<dcomp>& impurity_self_energy_down,
    std::vector<dcomp>& impurity_self_energy_lesser_up, std::vector<dcomp>& impurity_self_energy_lesser_down, double* spin_up, double* spin_down)
{
	get_spin_occupation(parameters, impurity_gf_up_lesser, impurity_gf_down_lesser, spin_up, spin_down);
	std::cout << std::setprecision(15) << "The spin up occupancy is " << *spin_up << "\n";
	std::cout << "The spin down occupancy is " << *spin_down << "\n";

	if (parameters.interaction_order == 2) {

        //one can choose a normal intergation (self_energy_2nd_order) or a krammer kronig method (self_energy_2nd_order_krammer_kronig)
		if (voltage_step == 0){
		self_energy_2nd_order_kramers_kronig(
		    parameters, impurity_gf_up, impurity_gf_down, impurity_gf_up_lesser, impurity_gf_down_lesser, impurity_self_energy_up, impurity_self_energy_lesser_up);
		self_energy_2nd_order_kramers_kronig(
		    parameters, impurity_gf_down, impurity_gf_up, impurity_gf_down_lesser, impurity_gf_up_lesser, impurity_self_energy_down, impurity_self_energy_lesser_down);
		} else {
			self_energy_2nd_order(
		    	parameters, impurity_gf_up, impurity_gf_down, impurity_gf_up_lesser, impurity_gf_down_lesser, impurity_self_energy_up, impurity_self_energy_lesser_up);
			self_energy_2nd_order(
			    parameters, impurity_gf_down, impurity_gf_up, impurity_gf_down_lesser, impurity_gf_up_lesser, impurity_self_energy_down, impurity_self_energy_lesser_down);
	
		}

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

void dmft(Parameters& parameters, int voltage_step, std::vector<std::vector<dcomp>>& self_energy_mb_up, std::vector<std::vector<dcomp>>& self_energy_mb_down,
    std::vector<std::vector<dcomp>>& self_energy_mb_lesser_up, std::vector<std::vector<dcomp>>& self_energy_mb_lesser_down, std::vector<Eigen::MatrixXcd>& gf_local_up,
    std::vector<Eigen::MatrixXcd>& gf_local_down, std::vector<Eigen::MatrixXcd>& gf_local_lesser_up, std::vector<Eigen::MatrixXcd>& gf_local_lesser_down,
    std::vector<std::vector<EmbeddingSelfEnergy>>& leads, std::vector<double>& spins_occup, const std::vector<std::vector<Eigen::MatrixXd>>& hamiltonian)
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
			    impurity_self_energy_down, impurity_self_energy_lesser_up, impurity_self_energy_lesser_down, &spins_occup.at(i), &spins_occup.at(i + parameters.chain_length));

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
