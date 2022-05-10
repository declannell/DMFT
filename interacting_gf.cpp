#include "parameters.h"
#include "leads_self_energy.h"
#include "dmft.h"
#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "interacting_gf.h"
#include <limits>


double Interacting_GF::kx() const { return kx_value; }
double Interacting_GF::ky() const { return ky_value; }

Interacting_GF::Interacting_GF(Parameters &parameters, double _kx, double _ky, std::vector<std::vector<dcomp>> &self_energy_mb, std::vector<dcomp> &self_energy_left,
                std::vector<dcomp> &self_energy_right): kx_value(_kx), ky_value(_ky)
{
        std::cout << "here7";
    this->interacting_gf.resize(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
    Eigen::MatrixXcd hamiltonian = get_hamiltonian(parameters);
    get_interacting_gf(parameters, hamiltonian, self_energy_mb, self_energy_left,
                                              self_energy_right);
}

Eigen::MatrixXcd Interacting_GF::get_hamiltonian(Parameters const &parameters){
    Eigen::MatrixXcd hamiltonian = Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length);
    for (int i = 0; i < parameters.chain_length - 1; i++){
        hamiltonian(i, i + 1) = parameters.hopping;
        hamiltonian(i + 1, i) = parameters.hopping;
    }

    for (int i = 0; i < parameters.chain_length; i++){
        double potential_bias = (parameters.voltage_l[parameters.voltage_step] -
                             parameters.voltage_r[parameters.voltage_step]);
        double voltage_i = parameters.voltage_l[parameters.voltage_step] - (double)(i + 1) / (double)(parameters.chain_length + 1.0) * potential_bias;

        hamiltonian(i, i) = parameters.onsite + 2 * parameters.hopping_x * cos(this->kx()) + 2 * parameters.hopping_y * \
                cos(this->ky()) + voltage_i;
    }
    /*
    std::cout << "The hamiltonian is " <<  std::endl;
    std::cout << hamiltonian << std::endl;
    std::cout << std::endl;
    */
    return hamiltonian;
}


void Interacting_GF::get_interacting_gf(Parameters &parameters, const Eigen::MatrixXcd& hamiltonian, std::vector<std::vector<dcomp>>  const &self_energy_mb, 
    std::vector<dcomp> const &self_energy_left, std::vector<dcomp> const &self_energy_right){
    std::cout << "here1";
    Eigen::MatrixXcd inverse_gf;
    for(int r = 0; r < parameters.steps; r++){
        inverse_gf = Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length);
        if (parameters.chain_length != 1){
            inverse_gf(0, 0) = parameters.energy.at(r) - hamiltonian(0, 0) - self_energy_mb.at(0).at(r) - self_energy_left.at(r);

            inverse_gf(parameters.chain_length - 1, parameters.chain_length - 1) = parameters.energy.at(r) - 
                hamiltonian(parameters.chain_length - 1, parameters.chain_length - 1) - self_energy_mb.at(parameters.chain_length - 1).at(r) - self_energy_right.at(r);

        } else if (parameters.chain_length == 1) {
            inverse_gf(0, 0) = parameters.energy.at(r) - hamiltonian(0, 0) - self_energy_mb.at(0).at(r) - self_energy_left.at(r) - self_energy_right.at(r);
        }

        for(int i = 0; i < parameters.chain_length; i++){
            for(int j = 0; j < parameters.chain_length; j++){
                if (i == j && ((i != 0) && (i != parameters.chain_length - 1))) {
                    //std::cout << i  << j << std::endl;
                    inverse_gf(i, i) = parameters.energy.at(r) - hamiltonian(i, i) - self_energy_mb.at(i).at(r);
                } else if (i != j) {
                    inverse_gf(i, j) = - hamiltonian(i, j);
                }
            }           
        }
    this->interacting_gf.at(r) = inverse_gf.inverse();
    //std::cout << "The inverse of A is:\n" << interacting_gf.at(r)(0, 0) << std::endl;

    }
}

void get_analytic_gf_1_site(Parameters &parameters, std::vector<Eigen::MatrixXcd> green_function){
    std::vector<dcomp> analytic_gf(parameters.steps);

    EmbeddingSelfEnergy leads(parameters, M_PI / 2.0, M_PI / 2.0);
    double difference = -std::numeric_limits<double>::infinity();
    
    for(int r = 0; r < parameters.steps; r++){
        double x = parameters.energy.at(r).real() - parameters.onsite - 2.0 * parameters.hopping_x * cos(M_PI / 2.0) 
                    - 2.0 * parameters.hopping_y * cos(M_PI / 2.0) - leads.self_energy_left.at(r).real() - 
                    leads.self_energy_right.at(r).real();

        double y = leads.self_energy_left.at(r).imag() + 
                    leads.self_energy_right.at(r).imag();

        
        analytic_gf.at(r) = x / (x * x + y * y) +  parameters.j1 * y / (x * x + y * y);

        double real_difference = abs(analytic_gf.at(r).real() - green_function.at(r)(0, 0).real());
        double imag_difference = abs(analytic_gf.at(r).imag() - green_function.at(r)(0, 0).imag());
        if (real_difference > 0.001 || imag_difference > 0.001){
            std::cout << analytic_gf.at(r) << " " << green_function.at(r)(0, 0) << " " <<  r << "\n";         
        }
        difference = std::max(difference, std::max(real_difference, imag_difference));
    }
    std::cout << "The difference between the numerical and the analytic greeen function is " << difference << std::endl;
    std::cout << parameters.j1;
    std::ofstream myfile;
    myfile.open("/home/declan/green_function_code/quantum_transport/textfiles/gf_c++_analytic.txt");
    // myfile << parameters.steps << std::endl;
    for (int r = 0; r < parameters.steps; r++)
    {
        myfile << analytic_gf.at(r).real() << "," << analytic_gf.at(r).imag() << "\n";
        // std::cout << leads.self_energy_left.at(r) << "\n";
    }

    myfile.close();
}

void get_local_gf(Parameters &parameters, std::vector<double> const &kx, std::vector<double> const &ky, std::vector<std::vector<dcomp>> &self_energy_mb, 
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &gf_local_down){

    double num_k_points = parameters.chain_length_x * parameters.chain_length_y;
    std::cout << num_k_points << "\n";
    for(int kx_i = 0; kx_i < parameters.chain_length_x; kx_i++) {
        for(int ky_i = 0; ky_i < parameters.chain_length_y; ky_i++) {

            Interacting_GF gf_interacting_up(parameters,
                kx.at(kx_i), ky.at(ky_i), self_energy_mb,
                leads.at(kx_i).at(ky_i).self_energy_left,
                leads.at(kx_i).at(ky_i).self_energy_right);

            Interacting_GF gf_interacting_down(parameters,
                kx.at(kx_i), ky.at(ky_i), self_energy_mb,
                leads.at(kx_i).at(ky_i).self_energy_left,
                leads.at(kx_i).at(ky_i).self_energy_right);             

            for(int r = 0; r < parameters.steps; r++){
                for(int i = 0; i < parameters.chain_length; i++){
                    for(int j = 0; j < parameters.chain_length; j++){
                        gf_local_up.at(r)(i, j) += gf_interacting_up.interacting_gf.at(r)(i, j) / num_k_points;
                        gf_local_down.at(r)(i, j) += gf_interacting_down.interacting_gf.at(r)(i, j) / num_k_points;
                    }
                }
            }
        }
    }
}
