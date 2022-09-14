#include "dmft.h"
#include <mpi.h>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <vector>

#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"
#include "AIM.h"

void get_spin_occupation(const Parameters &parameters, const std::vector<double> &gf_lesser_up,
                        const std::vector<double> &gf_lesser_down, double *spin_up, double *spin_down)
{
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)(parameters.steps);
	double result_up = 0.0, result_down = 0.0;

	for (int r = 0; r < parameters.steps; r++) {
		//std::cout << parameters.energy.at(r) << " " << gf_lesser_down.at(r).imag() <<  std::endl;
		if (r == 0 || r == parameters.steps - 1) {
			result_up += (delta_energy / 2.0) * gf_lesser_up.at(r);
			result_down += (delta_energy / 2.0) * gf_lesser_down.at(r);
		} else {
			result_up += (delta_energy) * gf_lesser_up.at(r);
			result_down += (delta_energy) * gf_lesser_down.at(r);
		}
	}

	*spin_up = 1.0 / (2.0 * M_PI) * result_up;
	*spin_down = 1.0 / (2.0 * M_PI) * result_down;
}

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

double integrate(const Parameters& parameters, const std::vector<double>& gf_1, const std::vector<double>& gf_2, const std::vector<double>& gf_3, const int r)
{
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
	double result = 0;
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


double get_prefactor(const int i, const int j, const int r, const int voltage_step, const Parameters &parameters, AIM &aim_up, AIM &aim_down)
{
	double prefactor;
	if (voltage_step == 0){
		prefactor = fermi_function(parameters.energy.at(i), parameters) * fermi_function(parameters.energy.at(j), parameters) 
			+ (1 - fermi_function(parameters.energy.at(i), parameters) - fermi_function(parameters.energy.at(j), parameters)) *
			 fermi_function(parameters.energy.at(j) + parameters.energy.at(i) - parameters.energy.at(r), parameters); 
	} else {
		int a = i + j - r;
		dcomp prefactor_complex = aim_up.fermi_function_eff.at(i) * aim_down.fermi_function_eff.at(j) + 
			aim_down.fermi_function_eff.at(a) * (1.0 - aim_up.fermi_function_eff.at(i) - aim_down.fermi_function_eff.at(j));
		prefactor = prefactor_complex.real();
	}		
	return prefactor;
}

double integrate_equilibrium(const Parameters& parameters, const std::vector<double>& gf_1, const std::vector<double>& gf_2, const std::vector<double>& gf_3, const int r,
    AIM &aim_up, AIM &aim_down, int voltage_step)
{
	double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
	double result = 0;
	for (int i = 0; i < parameters.steps; i++) {
		for (int j = 0; j < parameters.steps; j++) {
			if (((i + j - r) > 0) && ((i + j - r) < parameters.steps)) {
				double prefactor = get_prefactor(i, j, r, voltage_step, parameters, aim_up, aim_down);
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

void self_energy_2nd_order_kramers_kronig(const Parameters& parameters, AIM &aim_up, AIM &aim_down, const int voltage_step)
{
	std::vector<double> impurity_self_energy_real(parameters.steps), impurity_self_energy_imag(parameters.steps);

	std::vector<double> impurity_gf_down_advanced_imag(parameters.steps), gf_greater_down(parameters.steps);
	std::vector<double> impurity_gf_up_imag(parameters.steps), impurity_gf_down_imag(parameters.steps);

	for (int r = 0; r < parameters.steps; r++) {
		gf_greater_down.at(r) = aim_down.dynamical_field_lesser.at(r) + (aim_down.dynamical_field_retarded.at(r) - std::conj(aim_down.dynamical_field_retarded.at(r))).imag();
		impurity_gf_up_imag.at(r) = aim_up.dynamical_field_retarded.at(r).imag();
		impurity_gf_down_imag.at(r) = aim_down.dynamical_field_retarded.at(r).imag();
    }
    //I only want to calculate the imaginary part of the self energy.

	for (int r = 0; r < parameters.steps; r++){
		impurity_self_energy_imag.at(r) = parameters.hubbard_interaction * parameters.hubbard_interaction
		    * integrate_equilibrium(parameters, impurity_gf_up_imag, impurity_gf_down_imag, impurity_gf_down_imag, r, aim_up, aim_down, voltage_step); 	

		aim_up.self_energy_mb_lesser.at(r) =
		     - parameters.hubbard_interaction * parameters.hubbard_interaction * (integrate(parameters, aim_up.dynamical_field_lesser,
			aim_down.dynamical_field_lesser, gf_greater_down, r)); //the minus is here causes the lesser GF is real

		//std::cout << aim_up.self_energy_mb_lesser.at(r) << std::endl;
	}
	//std::ostringstream ossser;
	//ossser << "textfiles/"
	//       << "se_krammer_kronig.txt";
	//std::string var = ossser.str();
	//std::ofstream se_krammer_kronig;
	//se_krammer_kronig.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		impurity_self_energy_real.at(r) = kramer_kronig_relation(parameters, impurity_self_energy_imag, r);
		aim_up.self_energy_mb_retarded.at(r) = impurity_self_energy_real.at(r) + parameters.j1 * impurity_self_energy_imag.at(r);

		//se_krammer_kronig << parameters.energy.at(r) << "  " << aim_up.self_energy_mb_retarded.at(r).real() << "  " << aim_up.self_energy_mb_retarded.at(r).imag() << "\n";
	}
	//se_krammer_kronig.close();
}

void impurity_solver(const Parameters &parameters, const int voltage_step, 
    AIM &aim_up, AIM &aim_down, double *spin_up, double *spin_down)
{
	get_spin_occupation(parameters, aim_up.dynamical_field_lesser, aim_down.dynamical_field_lesser, spin_up, spin_down);
	std::cout << std::setprecision(15) << "The spin up occupancy is " << *spin_up << "\n";
	std::cout << "The spin down occupancy is " << *spin_down << "\n";

	if (parameters.interaction_order == 2) {

        //one can choose a normal intergation (self_energy_2nd_order) or a krammer kronig method (self_energy_2nd_order_krammer_kronig)
		self_energy_2nd_order_kramers_kronig(parameters, aim_up, aim_down, voltage_step);
		self_energy_2nd_order_kramers_kronig(parameters, aim_down, aim_up, voltage_step);

		for (int r = 0; r < parameters.steps; r++) {
			aim_up.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * (*spin_down);
			aim_down.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * (*spin_up);
			//std::cout << aim_up.self_energy_mb_retarded.at(r) << "\n";
		}
	}

	if (parameters.interaction_order == 1) {
		for (int r = 0; r < parameters.steps; r++) {
			aim_up.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * (*spin_down);
			aim_down.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * (*spin_up);
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

	std::vector<dcomp> diag_gf_local_up(parameters.steps), diag_gf_local_down(parameters.steps), diag_gf_local_lesser_up(parameters.steps),
	    diag_gf_local_lesser_down(parameters.steps), impurity_self_energy_up(parameters.steps), impurity_self_energy_down(parameters.steps),
	    impurity_self_energy_lesser_up(parameters.steps), impurity_self_energy_lesser_down(parameters.steps);

	while (difference > parameters.convergence && count < parameters.self_consistent_steps) {

		get_difference(parameters, gf_local_up, old_green_function, difference, index);
		std::cout << "The difference is " << difference << ". The count is " << count << std::endl;
		if (difference < parameters.convergence) {
			break;
		}

		for (int i = 0; i < parameters.chain_length; i++) {  //we only do the dmft loop over the correlated metal.
			if (parameters.atom_type.at(i) == 0){
				continue;
			}
			for (int r = 0; r < parameters.steps; r++) {
				diag_gf_local_up.at(r) = gf_local_up.at(r)(i, i);
				diag_gf_local_down.at(r) = gf_local_down.at(r)(i, i);
				diag_gf_local_lesser_up.at(r) = gf_local_lesser_up.at(r)(i, i);
				diag_gf_local_lesser_down.at(r) = gf_local_lesser_down.at(r)(i, i);
				impurity_self_energy_up.at(r) = self_energy_mb_up.at(i).at(r);
				impurity_self_energy_down.at(r) = self_energy_mb_down.at(i).at(r);
				impurity_self_energy_lesser_up.at(r) = self_energy_mb_lesser_up.at(i).at(r);
				impurity_self_energy_lesser_down.at(r) = self_energy_mb_lesser_down.at(i).at(r);
			}

			std::cout << "atom which we put on correlation" << i << std::endl;

    		AIM aim_up(parameters, diag_gf_local_up, diag_gf_local_lesser_up, impurity_self_energy_up, impurity_self_energy_lesser_up, voltage_step);
    		AIM aim_down(parameters, diag_gf_local_down, diag_gf_local_lesser_down, impurity_self_energy_down, impurity_self_energy_lesser_down, voltage_step);
			
			std::cout << "AIM was created" << std::endl;
			impurity_solver(parameters, voltage_step, aim_up, aim_down, &spins_occup.at(i), &spins_occup.at(i + parameters.chain_length));

            if(count == 0){
                for (int r = 0; r < parameters.steps; r++) {
                    self_energy_mb_up.at(i).at(r) = aim_up.self_energy_mb_retarded.at(r);
					//std::cout << self_energy_mb_up.at(i).at(r) << " " << aim_up.self_energy_mb_retarded.at(r) << "\n";
                    self_energy_mb_down.at(i).at(r) = aim_down.self_energy_mb_retarded.at(r);
                    self_energy_mb_lesser_up.at(i).at(r) = parameters.j1 * aim_up.self_energy_mb_lesser.at(r);
                    self_energy_mb_lesser_down.at(i).at(r) = parameters.j1 * aim_down.self_energy_mb_lesser.at(r);
                }
            } else {
                for (int r = 0; r < parameters.steps; r++) {
                    self_energy_mb_up.at(i).at(r) = (aim_up.self_energy_mb_retarded.at(r) + self_energy_mb_up.at(i).at(r)) * 0.5;
                    self_energy_mb_down.at(i).at(r) = (aim_down.self_energy_mb_retarded.at(r) + self_energy_mb_down.at(i).at(r)) * 0.5;
                    self_energy_mb_lesser_up.at(i).at(r) = (parameters.j1 * aim_up.self_energy_mb_lesser.at(r) + self_energy_mb_lesser_up.at(i).at(r)) * 0.5;
                    self_energy_mb_lesser_down.at(i).at(r) = (parameters.j1 * aim_down.self_energy_mb_lesser.at(r) + self_energy_mb_lesser_down.at(i).at(r)) * 0.5;
                }
            }
		}
		
		get_local_gf_r_and_lesser(parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, gf_local_up, gf_local_lesser_up, voltage_step, hamiltonian);
		get_local_gf_r_and_lesser(parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, gf_local_down, gf_local_lesser_down, voltage_step, hamiltonian);
		count++;
	}
}
