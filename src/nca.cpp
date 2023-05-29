#include "parameters.h"
#include "green_function.h"
#include "impurity_solver.h"
#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
#include <limits>

void get_greater_lesser_se_fermion(const Parameters &parameters, const Interacting_GF &boson, Interacting_GF &fermion, int voltage_step) {
    
    fermion.lesser_se.clear();
    fermion.greater_se.clear();

    fermion.lesser_se.resize(parameters.steps, 0);
    fermion.greater_se.resize(parameters.steps, 0);

    
    //this uses the WBA 
    for (int r = 0; r < parameters.steps; r++) {
        //std::cout <<  boson.lesser_gf.at(r) << std::endl;
        for (int i = 0; i < parameters.steps; i++) {

            double coupling = parameters.gamma * parameters.bandwidth * parameters.bandwidth /
             ((parameters.energy.at(r) - parameters.energy.at(i)) * (parameters.energy.at(r) - parameters.energy.at(i)) + parameters.bandwidth * parameters.bandwidth);
            //if (r == 0) {
            //    std::cout << parameters.energy.at(r)  << " " << parameters.energy.at(i) << " " << fermi_function(parameters.energy.at(r) - parameters.energy.at(i) - parameters.voltage_l[voltage_step], parameters) << std::endl;
            //}
            fermion.greater_se.at(r) += 1.0 / (2 * M_PI) * coupling * boson.greater_gf.at(i) * (2.0 - 
                fermi_function(parameters.energy.at(r) - parameters.energy.at(i) - parameters.voltage_l[voltage_step], parameters) - 
                fermi_function(parameters.energy.at(r) - parameters.energy.at(i) - parameters.voltage_r[voltage_step], parameters)) * parameters.delta_energy;

            fermion.lesser_se.at(r) -= 1.0 / (2 * M_PI) * coupling * boson.lesser_gf.at(i) * ( 
                fermi_function(parameters.energy.at(r) - parameters.energy.at(i) - parameters.voltage_l[voltage_step], parameters) + 
                fermi_function(parameters.energy.at(r) - parameters.energy.at(i) - parameters.voltage_r[voltage_step], parameters)) * parameters.delta_energy;

            //fermion.lesser_se.at(r) += - 1.0 / (2 * M_PI) * parameters.gamma * boson.lesser_gf.at(i) *  
            //    (fermi_function(parameters.energy.at(r) - parameters.energy.at(i) - parameters.voltage_l[voltage_step], parameters) +
            //    fermi_function(parameters.energy.at(r) - parameters.energy.at(i) - parameters.voltage_r[voltage_step], parameters)) * parameters.delta_energy;
        }

    //std::cout << parameters.energy.at(r) << " " <<  fermion.lesser_se.at(r) << " " << fermion.greater_se.at(r) << "\n";
    }

}

void get_retarded_se(const Parameters &parameters, Interacting_GF &green_function) {
    
    green_function.retarded_se.clear();
    green_function.retarded_se.resize(parameters.steps, 0);  

    for (int r = 0; r < parameters.steps; r++) {
        green_function.retarded_se.at(r) = 0.5 * green_function.greater_se.at(r);
    }

    for (int r = 0; r < parameters.steps; r++) {
        double se_real = 0;
        for (int i = 0; i < parameters.steps; i++) {
            if (i != r) {
                se_real += green_function.retarded_se.at(i).imag() / (parameters.energy.at(i) - parameters.energy.at(r));
            }
        }
        green_function.retarded_se.at(r) += se_real * parameters.delta_energy / M_PI;
    }

    //for (int r = 0; r < parameters.steps; r++) {
    //    for (int i = 0; i < parameters.steps; i++) {
    //        dcomp term = parameters.energy.at(r) - parameters.energy.at(i) + parameters.delta_gf * parameters.j1;
    //        green_function.retarded_se.at(r) += green_function.greater_se.at(i) / term;
    //    }
    //    green_function.retarded_se.at(r) = green_function.retarded_se.at(r) * parameters.j1 * parameters.delta_energy / (2 * M_PI);
    //}
}

void get_retarded_gf_fermion(const Parameters &parameters, Interacting_GF &green_function) {   
    for( int r = 0; r < parameters.steps; r++) {
        green_function.retarded_gf.at(r) = 1.0 / (parameters.energy.at(r) - parameters.onsite_cor - green_function.retarded_se.at(r));
    }
}

void get_greater_lesser_gf(const Parameters &parameters, Interacting_GF &green_function) {
    for (int r = 0; r < parameters.steps; r++) {
        green_function.greater_gf.at(r) = green_function.retarded_gf.at(r) * green_function.greater_se.at(r) * std::conj(green_function.retarded_gf.at(r));
        green_function.lesser_gf.at(r) = green_function.retarded_gf.at(r) * green_function.lesser_se.at(r) * std::conj(green_function.retarded_gf.at(r));
    }
}

void get_greater_lesser_gf(const Parameters &parameters, Interacting_GF &green_function, int count) {
    if (count < 10) {
        for (int r = 0; r < parameters.steps; r++) {
            green_function.greater_gf.at(r) = 0.5 * green_function.greater_gf.at(r) + 0.5 * green_function.retarded_gf.at(r) * green_function.greater_se.at(r) * std::conj(green_function.retarded_gf.at(r));
            green_function.lesser_gf.at(r) = 0.5 * green_function.lesser_gf.at(r) + 0.5 * green_function.retarded_gf.at(r) * green_function.lesser_se.at(r) * std::conj(green_function.retarded_gf.at(r));
        }
    } else if (count > 10 && count < 0) {
        for (int r = 0; r < parameters.steps; r++) {
            green_function.greater_gf.at(r) = 0.75 * green_function.greater_gf.at(r) + 0.25 * green_function.retarded_gf.at(r) * green_function.greater_se.at(r) * std::conj(green_function.retarded_gf.at(r));
            green_function.lesser_gf.at(r) = 0.75 * green_function.lesser_gf.at(r) + 0.25 * green_function.retarded_gf.at(r) * green_function.lesser_se.at(r) * std::conj(green_function.retarded_gf.at(r));
        }    
    } else {
        for (int r = 0; r < parameters.steps; r++) {
            green_function.greater_gf.at(r) = 0.9 * green_function.greater_gf.at(r) + 0.1 * green_function.retarded_gf.at(r) * green_function.greater_se.at(r) * std::conj(green_function.retarded_gf.at(r));
            green_function.lesser_gf.at(r) = 0.9 * green_function.lesser_gf.at(r) + 0.1 * green_function.retarded_gf.at(r) * green_function.lesser_se.at(r) * std::conj(green_function.retarded_gf.at(r));
        }
    }
}

void get_greater_lesser_se_boson(const Parameters &parameters, Interacting_GF &boson, const Interacting_GF &fermion_up,
    const Interacting_GF &fermion_down, int voltage_step) {

    boson.lesser_se.clear();
    boson.greater_se.clear();

    boson.lesser_se.resize(parameters.steps, 0);
    boson.greater_se.resize(parameters.steps, 0);

    //std::cout << "make sure the spin summation is correct in the spin non-polarised case. \n";

    for (int r = 0; r < parameters.steps; r++) {
        //std::cout << boson.lesser_se.at(r) << std::endl;
        for (int i = 0; i < parameters.steps; i++) {

            double coupling = parameters.gamma * parameters.bandwidth * parameters.bandwidth /
             ((parameters.energy.at(i) - parameters.energy.at(r))* (parameters.energy.at(i) - parameters.energy.at(r)) + parameters.bandwidth * parameters.bandwidth);

            boson.greater_se.at(r) += (1.0 / (2.0 * M_PI)) * coupling * parameters.delta_energy * (fermion_up.greater_gf.at(i) + fermion_down.greater_gf.at(i))
                * (fermi_function(parameters.energy.at(i) - parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) + 
                   fermi_function(parameters.energy.at(i) - parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters));

            boson.lesser_se.at(r) -= (1.0 / (2.0 * M_PI)) * parameters.delta_energy * coupling * (fermion_up.lesser_gf.at(i) + fermion_down.lesser_gf.at(i)) *
                (2.0 - fermi_function(parameters.energy.at(i) - parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) 
                    -  fermi_function(parameters.energy.at(i) - parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters));
            //if (r == 1) {
            //    std::cout << fermion_up.lesser_gf.at(i) << " " << 2.0 - fermi_function(parameters.energy.at(r) - parameters.energy.at(i) - parameters.voltage_l[voltage_step], parameters) 
            //        - fermi_function(parameters.energy.at(r) - parameters.energy.at(i) - parameters.voltage_r[voltage_step], parameters) << std::endl;
            //}
        }
    }
}

void get_retarded_gf_boson(const Parameters &parameters, Interacting_GF &green_function) {   
    for( int r = 0; r < parameters.steps; r++) {
        green_function.retarded_gf.at(r) = 1.0 / (parameters.energy.at(r) - green_function.retarded_se.at(r));
    }
}

double absolute_value(double num1) {
	return std::sqrt((num1 ) * (num1));
}

void get_difference(const Parameters &parameters, std::vector<dcomp> &old_self_energy, std::vector<dcomp> &new_self_energy,
                double &difference){
    difference = 0;
	double old_difference = 0;
	double real_difference = 0, imag_difference = 0;
	for (int r = 0; r < parameters.steps; r++) {
		real_difference = absolute_value(old_self_energy.at(r).real() - new_self_energy.at(r).real());
		imag_difference = absolute_value(old_self_energy.at(r).imag() - new_self_energy.at(r).imag());
		difference = std::max(difference, std::max(real_difference, imag_difference));
		old_self_energy.at(r) = new_self_energy.at(r);
    }
}
void intialise_gf(const Parameters &parameters, Interacting_GF &boson, Interacting_GF &fermion_up, Interacting_GF &fermion_down) {
    int a = parameters.steps / 3;
    for (int r = a; r < a *2; r++) {
        //fermion_up.lesser_gf.at(r) = parameters.j1;
        //fermion_down.lesser_gf.at(r) = parameters.j1;
        boson.lesser_gf.at(r) = - 1.0 * parameters.j1;
        //fermion_up.greater_gf.at(r) = - parameters.j1;
        //fermion_down.greater_gf.at(r) = - parameters.j1;
        boson.greater_gf.at(r) =  - 1.0 * parameters.j1;
    }
}

double get_z_prefactor(const Parameters &parameters, Interacting_GF &boson, Interacting_GF &fermion_up, Interacting_GF &fermion_down) {
    double z_prefactor = 0;

    for (int r = 0; r < parameters.steps; r++) {
        z_prefactor += - boson.lesser_gf.at(r).imag() + fermion_up.lesser_gf.at(r).imag() + fermion_down.lesser_gf.at(r).imag();
    }

    return  parameters.delta_energy * z_prefactor / (2.0 * M_PI);
}

void test_retarded_gf(const Parameters &parameters, Interacting_GF &boson, Interacting_GF &fermion_up, Interacting_GF &fermion_down) {

    double test_boson = 0.0, test_fermion_up = 0.0, test_fermion_down = 0.0;

    for (int r = 0; r < parameters.steps; r++) {
        test_boson += boson.retarded_gf.at(r).imag();
        test_fermion_up += fermion_up.retarded_gf.at(r).imag();
        test_fermion_down += fermion_down.retarded_gf.at(r).imag();
    }

    std::cout << "The imaginary part of the boson retarded gf integrates to " <<  test_boson * -1.0 * parameters.delta_energy / (M_PI) << "\n";
    std::cout << "The imaginary part of the fermion up retarded gf integrates to " <<  test_fermion_up * -1.0 * parameters.delta_energy / (M_PI) << "\n";
    std::cout << "The imaginary part of the fermion down retarded gf integrates to " <<  test_fermion_down * -1.0 * parameters.delta_energy / (M_PI) << "\n";
}

void impurity_solver(const Parameters &parameters, Interacting_GF &boson, Interacting_GF &fermion_up, Interacting_GF &fermion_down, int voltage_step,
     double &z_prefactor) {

    std::vector<dcomp> old_self_energy(parameters.steps, 0);
    int count = 0;
    double difference = std::numeric_limits<double>::infinity();
    std::vector<double> coupling;

    intialise_gf(parameters, boson, fermion_up, fermion_down);

    while (difference > parameters.convergence && count < parameters.self_consistent_steps) {
        if (parameters.spin_polarised == 1) {
            
            get_greater_lesser_se_fermion(parameters, boson, fermion_up, voltage_step);
            get_greater_lesser_se_fermion(parameters, boson, fermion_down, voltage_step);

            get_retarded_se(parameters, fermion_up);
            get_retarded_se(parameters, fermion_down);

            get_retarded_gf_fermion(parameters, fermion_up);
            get_retarded_gf_fermion(parameters, fermion_down);

            get_greater_lesser_gf(parameters, fermion_up);
            get_greater_lesser_gf(parameters, fermion_down);

            get_greater_lesser_se_boson(parameters, boson, fermion_up, fermion_down, voltage_step);

            get_retarded_se(parameters, boson);
            
            get_retarded_gf_boson(parameters, boson);
            
            get_greater_lesser_gf(parameters, boson, count);
        }

        get_difference(parameters, old_self_energy, fermion_up.retarded_se, difference);
        count++;
        std::cout << "The count is " << count << ". The difference is " << difference << std::endl;
    }

    test_retarded_gf(parameters, boson, fermion_up, fermion_down);


    z_prefactor =  1.0 / get_z_prefactor(parameters, boson, fermion_up, fermion_down);
}

impurity_solver_nca(parameters, voltage_step, aim_up, aim_down);

#include "parameters.h"
#include "impurity_solver.h"
#include "green_function.h"
#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
#include <limits>
#include <iomanip>  

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

void print_gf_lesser_greater(const Parameters &parameters, const int &voltage_step, std::vector<dcomp> &gf_lesser, std::vector<dcomp> &gf_greater) {
	
	std::ostringstream ossgf;
	ossgf << voltage_step << ".greater_gf.dat";
	std::string var = ossgf.str();
	std::ofstream gf_greater_file;
	gf_greater_file.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		gf_greater_file << parameters.energy.at(r) << "  " << gf_greater.at(r).real() << "   " << gf_greater.at(r).imag() << "  \n";
	}
	gf_greater_file.close();

    ossgf.str("");
    ossgf.clear();
	ossgf << voltage_step << ".lesser_gf.dat";
	var = ossgf.str();
	std::ofstream gf_lesser_file;
	gf_lesser_file.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		gf_lesser_file << parameters.energy.at(r) << "  " << gf_lesser.at(r).real() << "   " << gf_lesser.at(r).imag() << "  \n";
	}
	gf_lesser_file.close();

}

void print_dos(const Parameters &parameters, const int &voltage_step, std::vector<dcomp> &gf_lesser, std::vector<dcomp> &gf_greater) {
	std::ostringstream ossgf;
	ossgf << voltage_step << ".dos.dat";
	double dos_integral = 0.0;
	std::string var = ossgf.str();
	std::ofstream dos;
	dos.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		dos << parameters.energy.at(r) << "  " << (gf_lesser.at(r).imag() - gf_greater.at(r).imag()) / (2.0 * M_PI) << "  \n";
		dos_integral += (gf_lesser.at(r).imag() - gf_greater.at(r).imag()) / (2.0 * M_PI);
	}
	std::cout << "The dos integrated for all energies is " << dos_integral * parameters.delta_energy << std::endl;
	dos.close();
}

void get_lesser_greater_gf(const Parameters &parameters, const Interacting_GF &boson, const Interacting_GF &fermion, 
     const double &z_prefactor, std::vector<dcomp> &gf_lesser, std::vector<dcomp> &gf_greater) {

	//std::cout << parameters.steps << std::endl;

	for (int r = 0; r < parameters.steps; r++) {
		for (int i = 0; i < parameters.steps; i++) {
			if (((i + r) >= (parameters.steps / 2)) && ((i + r) < 3 * (parameters.steps / 2))) {
				//std::cout << r << " " << i << "  " << parameters.energy.at(r) + parameters.energy.at(i)  << "  " << parameters.energy.at(r + i - (parameters.steps / 2)) << "\n ";
				gf_lesser.at(r) += boson.greater_gf.at(i) * fermion.lesser_gf.at(i + r - (parameters.steps / 2));
				gf_greater.at(r) += boson.lesser_gf.at(i) * fermion.greater_gf.at(i + r - (parameters.steps / 2));				
			}		
		}
		//std::cout << parameters.steps / 2 << std::endl;
		gf_lesser.at(r) = gf_lesser.at(r) * z_prefactor * parameters.j1 / (2.0 * M_PI) * parameters.delta_energy;
		gf_greater.at(r) = gf_greater.at(r) * z_prefactor * parameters.j1 / (2.0 * M_PI) * parameters.delta_energy;		
	}
}

void get_current(const Parameters &parameters, const std::vector<dcomp> &gf_lesser, const std::vector<dcomp> &gf_greater, const int voltage_step,
	 double &current_left, double &current_right) {
	for (int r = 0; r < parameters.steps; r++) {
		double hybridisation = (parameters.energy.at(r) * parameters.energy.at(r) + parameters.bandwidth * parameters.bandwidth);
		current_left += ((1.0 - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters)) * gf_lesser.at(r).imag() 
			+ fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) * gf_greater.at(r).imag()) / hybridisation;
		current_right += ((1.0 - fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters)) * gf_lesser.at(r).imag() 
			+ fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters) * gf_greater.at(r).imag()) / hybridisation;
	}
	current_left = current_left * parameters.delta_energy * parameters.gamma * parameters.bandwidth * parameters.bandwidth;
	current_right = current_right * parameters.delta_energy * parameters.gamma * parameters.bandwidth * parameters.bandwidth;	
}

void get_conductance(const Parameters &parameters, const int &voltage_step, std::vector<dcomp> &gf_lesser, std::vector<dcomp> &gf_greater) {
	double conductance = 0;

	for (int r = 0; r < parameters.steps; r++) {
		double dos = 0.5 * (gf_lesser.at(r).imag() - gf_greater.at(r).imag());
		double coupling = parameters.gamma * parameters.bandwidth * parameters.bandwidth / (parameters.energy.at(r) * parameters.energy.at(r) + parameters.bandwidth * parameters.bandwidth);
		double derivative =  exp((parameters.energy.at(r) - parameters.chemical_potential) / parameters.temperature) / (parameters.temperature * 
		(1.0 + exp((parameters.energy.at(r) - parameters.chemical_potential) / parameters.temperature)) *
		(1.0 + exp((parameters.energy.at(r) - parameters.chemical_potential) / parameters.temperature))); 
		if (std::isnan(derivative) == true) { //this is a big hack but i think it is just numerical problems that the derivative can be nan away from the chemical potential
			derivative = 0;
		}

		//std::cout << parameters.energy.at(r) << " " << dos << " " << coupling << " " << derivative << "\n";
		conductance += dos * coupling * derivative;
	}

	std::cout << "The conductance is " << conductance * parameters.delta_energy * 2.0 * M_PI << "\n";
}


void get_spin_occupation(const Parameters &parameters, const std::vector<dcomp> &gf_lesser, double &z_prefactor)
{
	double occupation = 0.0;

	for (int r = 0; r < parameters.steps; r++) {
		occupation += gf_lesser.at(r).imag();
	}
	
	occupation = occupation * parameters.delta_energy * z_prefactor / (2.0 * M_PI);
	std::cout << "the pseudo fermion occupation is " << occupation << std::endl;
}

void get_spin_occupation(const Parameters &parameters, const std::vector<dcomp> &gf_lesser)
{
	double occupation = 0.0;

	for (int r = 0; r < parameters.steps; r++) {
		occupation += gf_lesser.at(r).imag();
	}
	
	occupation = occupation * parameters.delta_energy / (2.0 * M_PI);
	std::cout << "the occupation is " << occupation << std::endl;
}

void impurity_solver_nca(const Parameters &parameters, const int voltage_step, AIM &aim_up, AIM &aim_down)
{
	Interacting_GF boson(parameters);
	Interacting_GF fermion_up(parameters);
	Interacting_GF fermion_down(parameters);	

	double z_prefactor = 0;
	double up_occupation = parameters.spin_up_occup, down_occupation = parameters.spin_down_occup;

	impurity_solver(parameters, boson, fermion_up, fermion_down, voltage_step, z_prefactor);
	//std::cout << "The ratio of Z_0 / Z_1 is " << z_prefactor << std::endl;
	get_lesser_greater_gf(parameters, boson, fermion_up, z_prefactor, aim_up.impurity_gf_mb_lesser, gf_greater_up);
	get_lesser_greater_gf(parameters, boson, fermion_down, z_prefactor, gf_lesser_down, gf_greater_down);

	//boson.print_green_function(parameters, m, "boson");
	//fermion_up.print_green_function(parameters, m, "fermion_up");
	//fermion_down.print_green_function(parameters, m, "fermion_down");

	//print_gf_lesser_greater(parameters, voltage_step, gf_lesser_up, gf_greater_up);
	print_dos(parameters, m, gf_lesser_up, gf_greater_up);
	get_spin_occupation(parameters, fermion_up.lesser_gf, z_prefactor);
	if (m == 0) {
		get_conductance(parameters, m, gf_lesser_up, gf_greater_up);
	}
	get_current(parameters, gf_lesser_up, gf_greater_up, m, current_left.at(m), current_right.at(m));
	get_spin_occupation(parameters, gf_lesser_up);
}