#include <mpi.h>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include "dmft.h"
#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"
#include "transport.h"
#include "analytic_gf.h"

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void analytic_gf(Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local){
    std::vector<Eigen::MatrixXcd> gf_analytic(parameters.steps, Eigen::MatrixXcd::Zero(2, 2));
    std::vector<dcomp> spectral(parameters.steps, 0);

    double diff = 0.0;
    for (int r = 0; r < parameters.steps; r++){
        double a = parameters.energy.at(r) - parameters.onsite_cor;
        double b =  a * a - parameters.gamma * parameters.gamma - parameters.hopping_cor * parameters.hopping_cor;

        double demoninator = b * b + 4 * parameters.gamma * parameters.gamma * a * a;
        double real_gf = (a * b + 2  * parameters.gamma * parameters.gamma * a) / (demoninator);
        double imag_gf = (2 * parameters.gamma * a * a - parameters.gamma * b) / demoninator;

        
        gf_analytic.at(r)(0, 0) = real_gf + parameters.j1 * imag_gf;
        gf_analytic.at(r)(1, 1) = real_gf + parameters.j1 * imag_gf; 
        gf_analytic.at(r)(0, 1) = parameters.hopping_cor/ demoninator;
        gf_analytic.at(r)(1, 0) = parameters.hopping_cor/ demoninator;               

        if (abs(gf_local.at(r)(0, 0) - gf_analytic.at(r)(0, 0)) > diff){
            diff = abs(gf_local.at(r)(0, 0) - gf_analytic.at(r)(0, 0));
        }
        //std::cout << parameters.energy.at(r) << "  " << gf_analytic.at(r).real() << " " << gf_analytic.at(r).imag() << " "
		//               << gf_local.at(r)(0, 0).real() << " " << gf_local.at(r)(0, 0).imag() << " " << diff << "\n";
    }
	std::ofstream analytic_gf_file;
	analytic_gf_file.open("textfiles/analutic_gf.txt");
	for (int r = 0; r < parameters.steps; r++) {
		analytic_gf_file << parameters.energy.at(r) << "  " << gf_analytic.at(r)(0, 0).real() << " " << gf_analytic.at(r)(0, 0).imag() << " "
		               << gf_local.at(r)(0, 0).real() << " " << gf_local.at(r)(0, 0).imag() << "\n";
	}
		analytic_gf_file.close();
    std::cout << "The largest difference between the analytic noninteracting gf and the numerical noninteracting gf is " << diff << std::endl;

}


/*
void analytic_gf(Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local, std::vector<dcomp> const &self_energy_left, std::vector<dcomp> const &self_energy_right){
    std::vector<dcomp> gf_analytic(parameters.steps, 0);
    std::vector<dcomp> spectral(parameters.steps, 0);
    std::vector<dcomp> self_energy_analytic(parameters.steps, 0);
    std::vector<dcomp> coupling_left(parameters.steps, 0);
    double diff = 0.0;
    for (int r = 0; r < parameters.steps; r++){
        double x = (parameters.energy.at(r) - parameters.onsite_l) / (2.0 * abs(parameters.hopping_lz));
        if (abs(x) > 1){
            self_energy_analytic.at(r) = x - sgn(x) * sqrt(x * x - 1.0);
        } else {
            self_energy_analytic.at(r) = x - parameters.j1 * sqrt(1.0 - x * x);
        }

        self_energy_analytic.at(r) *= abs(parameters.hopping_lc);
        
        gf_analytic.at(r) = 1.0 / (parameters.energy.at(r) + parameters.delta_gf - parameters.onsite_cor 
                                   - 2.0 * self_energy_analytic.at(r));

        coupling_left.at(r) = parameters.j1 * (self_energy_analytic.at(r) - std::conj(self_energy_analytic.at(r)));



        dcomp coupling_right = coupling_left.at(r);

        double real = parameters.energy.at(r) - parameters.onsite_cor - (2.0 * self_energy_analytic.at(r)).real();

        spectral.at(r) = 2.0 * coupling_left.at(r) / (real * real + 0.25 * (coupling_left.at(r) + coupling_right) * (coupling_left.at(r) + coupling_right));


        if (abs(gf_local.at(r)(0, 0).real() - gf_analytic.at(r).real()) > diff){
            diff = abs(gf_local.at(r)(0, 0) - gf_analytic.at(r));
        }
        //std::cout << parameters.energy.at(r) << "  " << gf_analytic.at(r).real() << " " << gf_analytic.at(r).imag() << " "
		//               << gf_local.at(r)(0, 0).real() << " " << gf_local.at(r)(0, 0).imag() << " " << diff << "\n";
    }
	std::ofstream analytic_gf_file;
	analytic_gf_file.open("textfiles/analutic_gf.txt");
	for (int r = 0; r < parameters.steps; r++) {
		analytic_gf_file << parameters.energy.at(r) << "  " << gf_analytic.at(r).real() << " " << gf_analytic.at(r).imag() << " "
		               << gf_local.at(r)(0, 0).real() << " " << gf_local.at(r)(0, 0).imag() << " " << self_energy_analytic.at(r).real() << 
                       " " << self_energy_analytic.at(r).imag() << "  " << spectral.at(r).real() << " " << spectral.at(r).imag()<< " " << coupling_left.at(r).real() << "\n";
	}
		analytic_gf_file.close();
    std::cout << "The largest difference between the analytic noninteracting gf and the numerical noninteracting gf is " << diff << std::endl;

    integrate_spectral(parameters, self_energy_left, self_energy_right);

}

void integrate_spectral(Parameters &parameters, std::vector<dcomp> const &self_energy_left, std::vector<dcomp> const &self_energy_right){

	std::ofstream analytic_gf_file;
	analytic_gf_file.open("textfiles/analytic_gf.txt");
    std::vector<dcomp> spectral(parameters.steps, 0);
    std::vector<dcomp> self_energy_analytic(parameters.steps, 0);
    std::vector<dcomp> coupling_left(parameters.steps, 0);
    std::vector<dcomp> coupling_right(parameters.steps, 0);
    std::vector<dcomp> gf_analytic(parameters.steps, 0);

    double result = 0.0;


    double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;

    for (int r = 0; r < parameters.steps; r++){
        double x = (parameters.energy.at(r) - parameters.onsite_l) / (2.0 * abs(parameters.hopping_lz));
        
        gf_analytic.at(r) = 1.0 / (parameters.energy.at(r) + parameters.delta_gf - parameters.onsite_cor 
                                   - 2.0 * self_energy_analytic.at(r));

        if (abs(x) < 1){
            coupling_left.at(r) = 2.0 * abs(parameters.hopping_lc) * sqrt(1.0 - x * x);
            coupling_right.at(r) = 2.0 * abs(parameters.hopping_lc) * sqrt(1.0 - x * x);
            double real = - parameters.onsite_cor + parameters.onsite_l;
            double demoninator = real * real + (coupling_right.at(r) + coupling_left.at(r)).real() * (coupling_right.at(r) + coupling_left.at(r)).real() / 4.0;
            spectral.at(r) = (coupling_right.at(r) + coupling_left.at(r)).real() / demoninator;

        } else if (abs(x) >= 1) {
            double real = - parameters.onsite_cor + parameters.onsite_l- 2.0 * abs(parameters.hopping_cor) * (double)sgn(x) * sqrt(x * x - 1.0);
            double demoninator = real * real + (coupling_right.at(r) + coupling_left.at(r)).real() * (coupling_right.at(r) + coupling_left.at(r)).real() / 4.0;
            spectral.at(r) = (coupling_right.at(r) + coupling_left.at(r)).real() / demoninator;
        }
        result += spectral.at(r).real();
		analytic_gf_file << parameters.energy.at(r) << "  " << coupling_left.at(r).real() << " " << spectral.at(r).real() << "\n";
	}

    result *= delta_energy / (2.0 * M_PI);

    std::cout << "the value of the integral is " << result << std::endl;
	analytic_gf_file.close();

}
*/