
#include "dmft.h"
#include <mpi.h>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"
#include "aim.h"
#include "utilis.h"
#include "dmft.h"


void get_spin_occupation(const Parameters &parameters, const std::vector<double> &gf_lesser_up,
                        const std::vector<double> &gf_lesser_down, double *spin_up, double *spin_down)
{

	double result_up = 0.0, result_down = 0.0;

	if (parameters.spin_polarised == true) {
		for (int r = 0; r < parameters.steps_myid; r++) {
		//std::cout << parameters.energy.at(r) << " " << gf_lesser_down.at(r).imag() <<  std::endl;
			if (r + parameters.start.at(parameters.myid) == 0 || r + parameters.start.at(parameters.myid) == parameters.steps - 1) {
			//std::cout << "I am rank " << parameters.myid << std::endl;
				result_up += (parameters.delta_energy / 2.0) * gf_lesser_up.at(r);
				result_down += (parameters.delta_energy / 2.0) * gf_lesser_down.at(r);
			} else {
				result_up += (parameters.delta_energy) * gf_lesser_up.at(r);
				result_down += (parameters.delta_energy) * gf_lesser_down.at(r);
			}
		}
		result_up = 1.0 / (2.0 * M_PI) * result_up;
		result_down = 1.0 / (2.0 * M_PI) * result_down;
		MPI_Allreduce(&result_up, spin_up, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&result_down, spin_down, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
	} else {
		for (int r = 0; r < parameters.steps_myid; r++) {
			if (r + parameters.start.at(parameters.myid) == 0 || r + parameters.start.at(parameters.myid) == parameters.steps - 1) {
				result_up += (parameters.delta_energy / 2.0) * gf_lesser_up.at(r);
			} else {
				result_up += (parameters.delta_energy) * gf_lesser_up.at(r);
			}
		}
		result_up = 1.0 / (2.0 * M_PI) * result_up;
		MPI_Allreduce(&result_up, spin_up, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		*spin_down = *spin_up;
	}
}


dcomp integrate(const Parameters& parameters, const std::vector<dcomp>& gf_1, const std::vector<dcomp>& gf_2, const std::vector<dcomp>& gf_3, const int r)
{

	dcomp result = 0;
	for (int i = 0; i < parameters.steps; i++) {
		for (int j = 0; j < parameters.steps; j++) {
			if (((i + j - r) > 0) && ((i + j - r) < parameters.steps)) {
				//this integrates the equation in PHYSICAL REVIEW B 74, 155125 2006
				//I say the green function is zero outside e_lower_bound and e_upper_bound. This means I need the final green function in the integral to be within an energy of e_lower_bound
				//and e_upper_bound. The index of 0 corresponds to e_lower_bound. Hence we need i+J-r>0 but in order to be less an energy of e_upper_bound we need i+j-r<steps.
				//These conditions ensure the enrgy of the gf3 greens function to be within (e_upper_bound, e_lower_bound)
				result += (parameters.delta_energy / (2.0 * M_PI)) * (parameters.delta_energy / (2.0 * M_PI)) * gf_1.at(i) * gf_2.at(j) * gf_3.at(i + j - r);
			}
		}
	}
	return result;
}

void self_energy_2nd_order(const Parameters& parameters, AIM &aim_up, AIM &aim_down)
{

	std::vector<dcomp> gf_retarded_up(parameters.steps), gf_retarded_down(parameters.steps), advanced_down(parameters.steps), 
		gf_lesser_up(parameters.steps), gf_lesser_down(parameters.steps), gf_greater_down(parameters.steps);
	distribute_to_procs(parameters, gf_retarded_up, aim_up.dynamical_field_retarded);
	distribute_to_procs(parameters, gf_retarded_down, aim_down.dynamical_field_retarded);



	std::vector<dcomp> advanced_down_myid(parameters.steps_myid), gf_lesser_up_myid(parameters.steps_myid), gf_lesser_down_myid(parameters.steps_myid),
		gf_greater_down_myid(parameters.steps_myid);
	
	for (int r = 0; r < parameters.steps_myid; r++) {
		advanced_down_myid.at(r) = std::conj(aim_down.dynamical_field_retarded.at(r));
		gf_lesser_down_myid.at(r) = parameters.j1 * aim_down.dynamical_field_lesser.at(r);
		gf_lesser_up_myid.at(r) = parameters.j1 * aim_up.dynamical_field_lesser.at(r);
		gf_greater_down_myid.at(r) = parameters.j1 * aim_down.dynamical_field_lesser.at(r) + aim_down.dynamical_field_retarded.at(r) - std::conj(aim_down.dynamical_field_retarded.at(r));
	}

	distribute_to_procs(parameters, advanced_down, advanced_down_myid);
	distribute_to_procs(parameters, gf_lesser_down, gf_lesser_down_myid);
	distribute_to_procs(parameters, gf_lesser_up, gf_lesser_up_myid);
	distribute_to_procs(parameters, gf_greater_down, gf_greater_down_myid);

	for (int r = 0; r < parameters.steps_myid; r++){
		int y = r + parameters.start.at(parameters.myid);
		aim_up.self_energy_mb_retarded.at(r) = parameters.hubbard_interaction * parameters.hubbard_interaction
		    * integrate(parameters, gf_retarded_up, gf_retarded_down, gf_lesser_down, y); //this resets the self energy	
		aim_up.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction
		    * integrate(parameters, gf_retarded_up, gf_lesser_down, gf_lesser_down, y); 
		aim_up.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction
		    * integrate(parameters, gf_lesser_up, gf_retarded_down, gf_lesser_down, y); 
		aim_up.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction
		    * integrate(parameters, gf_lesser_up, gf_lesser_down, advanced_down, y); 

		aim_up.self_energy_mb_lesser.at(r) = (parameters.hubbard_interaction * parameters.hubbard_interaction
		    * integrate(parameters, gf_lesser_up, gf_lesser_down, gf_greater_down, y)).imag(); 	
	}
}



double get_prefactor(const int i, const int j, const int r, const int voltage_step, const Parameters &parameters,
	 std::vector<double> &fermi_up, std::vector<double> &fermi_down)
{
	double prefactor;
	if (voltage_step == 0){
		prefactor = fermi_function(parameters.energy.at(i), parameters) * fermi_function(parameters.energy.at(j), parameters) 
			+ (1 - fermi_function(parameters.energy.at(i), parameters) - fermi_function(parameters.energy.at(j), parameters)) *
			 fermi_function(parameters.energy.at(j) + parameters.energy.at(i) - parameters.energy.at(r), parameters); 
	} else {
		int a = i + j - r;
		dcomp prefactor_complex = fermi_up.at(i) * fermi_down.at(j) + 
			fermi_down.at(a) * (1.0 - fermi_up.at(i) - fermi_down.at(j));
		prefactor = prefactor_complex.real();
	}		
	return prefactor;
}

double integrate_equilibrium(const Parameters& parameters, const std::vector<double>& gf_1, const std::vector<double>& gf_2, const std::vector<double>& gf_3, const int r,
    std::vector<double> &fermi_up, std::vector<double> &fermi_down, int voltage_step)
{

	double result = 0;
	for (int i = 0; i < parameters.steps; i++) {
		for (int j = 0; j < parameters.steps; j++) {
			if (((i + j - r) > 0) && ((i + j - r) < parameters.steps)) {
				double prefactor = get_prefactor(i, j, r, voltage_step, parameters, fermi_up, fermi_down);
				//this integrates the equation in PHYSICAL REVIEW B 74, 155125 2006
				//I say the green function is zero outside e_lower_bound and e_upper_bound. This means I need the final green function in the integral to be within an energy of e_lower_bound
				//and e_upper_bound. The index of 0 corresponds to e_lower_bound. Hence we need i+J-r>0 but in order to be less an energy of e_upper_bound we need i+j-r<steps.
				//These conditions ensure the enrgy of the gf3 greens function to be within (e_upper_bound, e_lower_bound)
				result += prefactor * (parameters.delta_energy / (M_PI)) * (parameters.delta_energy / (M_PI)) * gf_1.at(i) * gf_2.at(j) * gf_3.at(i + j - r);
			}
		}
	}
	return result;
}

void self_energy_2nd_order_kramers_kronig(const Parameters& parameters, AIM &aim_up, AIM &aim_down, const int voltage_step)
{

	std::vector<double> impurity_self_energy_imag(parameters.steps); //this is for the kramer-kronig relation. 
	std::vector<double> impurity_self_energy_real_myid(parameters.steps_myid), impurity_self_energy_imag_myid(parameters.steps_myid);

	std::vector<double> fermi_eff_up(parameters.steps), fermi_eff_down(parameters.steps);
	distribute_to_procs(parameters, fermi_eff_up,  aim_up.fermi_function_eff); //this is for the fermi function is the kk self energy.
	distribute_to_procs(parameters, fermi_eff_down, aim_down.fermi_function_eff);

	std::vector<dcomp> gf_lesser_up(parameters.steps), gf_lesser_down(parameters.steps), gf_greater_down(parameters.steps); //this is to pass into the integrater 
		//for the self energy.
	std::vector<dcomp> gf_lesser_up_myid(parameters.steps_myid), gf_lesser_down_myid(parameters.steps_myid), gf_greater_down_myid(parameters.steps_myid);
	
	for (int r = 0; r < parameters.steps_myid; r++) {
		gf_lesser_down_myid.at(r) = parameters.j1 * aim_down.dynamical_field_lesser.at(r);
		gf_lesser_up_myid.at(r) = parameters.j1 * aim_up.dynamical_field_lesser.at(r);
		gf_greater_down_myid.at(r) = parameters.j1 * aim_down.dynamical_field_lesser.at(r) + aim_down.dynamical_field_retarded.at(r) - std::conj(aim_down.dynamical_field_retarded.at(r));
	}	

	distribute_to_procs(parameters, gf_lesser_up, gf_lesser_up_myid);
	distribute_to_procs(parameters, gf_lesser_down, gf_lesser_down_myid);
	distribute_to_procs(parameters, gf_greater_down, gf_greater_down_myid);	

	std::vector<double> impurity_gf_up_imag(parameters.steps), impurity_gf_down_imag(parameters.steps);
	std::vector<double> impurity_gf_up_imag_myid(parameters.steps_myid), impurity_gf_down_imag_myid(parameters.steps_myid);

	for (int r = 0; r < parameters.steps_myid; r++) {
		impurity_gf_up_imag_myid.at(r) = aim_up.dynamical_field_retarded.at(r).imag();
		impurity_gf_down_imag_myid.at(r) = aim_down.dynamical_field_retarded.at(r).imag();
    }

	distribute_to_procs(parameters, impurity_gf_up_imag, impurity_gf_up_imag_myid);
	distribute_to_procs(parameters, impurity_gf_down_imag, impurity_gf_down_imag_myid);	

	for (int r = 0; r < parameters.steps_myid; r++){
		int y = r + parameters.start.at(parameters.myid);
		impurity_self_energy_imag_myid.at(r) = parameters.hubbard_interaction * parameters.hubbard_interaction
		    * integrate_equilibrium(parameters, impurity_gf_up_imag, impurity_gf_down_imag, impurity_gf_down_imag, y, fermi_eff_up, fermi_eff_down, voltage_step); 	
		
		aim_up.self_energy_mb_lesser.at(r) = (parameters.hubbard_interaction * parameters.hubbard_interaction * (integrate(parameters, gf_lesser_up,
			gf_lesser_down, gf_greater_down, y))).imag(); 
	}

	MPI_Barrier(MPI_COMM_WORLD);
	distribute_to_procs(parameters, impurity_self_energy_imag, impurity_self_energy_imag_myid);

	for (int r = 0; r < parameters.steps_myid; r++) {
		int y = r + parameters.start.at(parameters.myid); 
		impurity_self_energy_real_myid.at(r) = kramer_kronig_relation(parameters, impurity_self_energy_imag, y);
		aim_up.self_energy_mb_retarded.at(r) = impurity_self_energy_real_myid.at(r) + parameters.j1 * impurity_self_energy_imag_myid.at(r);
	}
}




void impurity_solver_sigma_2(const Parameters &parameters, const int voltage_step, 
    AIM &aim_up, AIM &aim_down, double *spin_up, double *spin_down)
{
	get_spin_occupation(parameters, aim_up.dynamical_field_lesser, aim_down.dynamical_field_lesser, spin_up, spin_down);
	if (parameters.myid == 0) {
		std::cout << std::setprecision(15) << "The spin up occupancy is " << *spin_up << "\n";
		std::cout << "The spin down occupancy is " << *spin_down << "\n";
	}

	if (parameters.impurity_solver == 2 || parameters.impurity_solver == 5) {// kramer kronig relation.
		if (parameters.myid == 0) std::cout << "using the kramer-kronig relation for second order perturbation theory\n";
		if (parameters.spin_polarised == true) {
			self_energy_2nd_order_kramers_kronig(parameters, aim_up, aim_down, voltage_step);
			self_energy_2nd_order_kramers_kronig(parameters, aim_down, aim_up, voltage_step);
		} else { //only need to do this for spin up. Note aim_down = aim_up
			self_energy_2nd_order_kramers_kronig(parameters, aim_up, aim_down, voltage_step);
		}
	}

	if (parameters.impurity_solver == 1 || parameters.impurity_solver == 4) {//brute force sigma_2
		if (parameters.myid == 0) std::cout << "using the brute force method for second order perturbation theory\n";
		if (parameters.spin_polarised == true) {
			self_energy_2nd_order(parameters, aim_up, aim_down);
			self_energy_2nd_order(parameters, aim_down, aim_up);
		} else { //only need to do this for spin up. Note aim_down = aim_up
			self_energy_2nd_order(parameters, aim_up, aim_down);
		}
	}
	
	if (parameters.impurity_solver == 1 || parameters.impurity_solver == 2 || parameters.impurity_solver == 0) {//if mean field or either implementation of sigma_2 then add the first order term.
		if (parameters.spin_polarised == true) {
			
			for (int r = 0; r < parameters.steps_myid; r++) {
				aim_up.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * (*spin_down);
				aim_down.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * (*spin_up);
			} 
		} else {
			for (int r = 0; r < parameters.steps_myid; r++) {
				aim_up.self_energy_mb_retarded.at(r) += parameters.hubbard_interaction * (*spin_down);
			}	
		}
	}

	if (parameters.impurity_solver == 3) {
		
	}
}
