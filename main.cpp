#include "parameters.h"
#include "leads_self_energy.h"
#include <iostream>
#include <vector>
#include <x86_64-linux-gnu/mpich/mpi.h>
#include <eigen3/Eigen/Dense>
#include "dmft.h"
#include "interacting_gf.h"


int main() {
    Parameters parameters = Parameters::init();
    
    std::vector<double> kx(parameters.chain_length_x, 0); 
    std::vector<double> ky(parameters.chain_length_y, 0); 
    
    for(int i = 0; i < parameters.chain_length_x; i++) {
        if (parameters.chain_length_x != 1) {
            kx.at(i) = 2 * M_PI * i / parameters.chain_length_x;
        } else if (parameters.chain_length_x == 1) {
            kx.at(i) = M_PI / 2.0;
        }
    }

    for(int i = 0; i < parameters.chain_length_y; i++) {
        if (parameters.chain_length_y != 1) {
            ky.at(i) = 2 * M_PI * i / parameters.chain_length_y;
        } else if (parameters.chain_length_y == 1) {
            ky.at(i) = M_PI / 2.0;
        }
    }

    std::cout << "The voltage difference is " <<  parameters.voltage_l[parameters.voltage_step] - parameters.voltage_r[parameters.voltage_step] << std::endl;

    std::vector<Eigen::MatrixXcd> gf_local_up(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
    std::vector<Eigen::MatrixXcd> gf_local_down(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
    std::vector<std::vector<dcomp>> self_energy_mb_up(parameters.chain_length, std::vector<dcomp> (parameters.steps));
    std::vector<std::vector<dcomp>> self_energy_mb_down(parameters.chain_length, std::vector<dcomp> (parameters.steps));

    std::vector<std::vector<EmbeddingSelfEnergy>> leads;
    for (int i = 0; i < parameters.chain_length_x; i++)
    {
        std::vector<EmbeddingSelfEnergy> vy;
        for (int j = 0; j < parameters.chain_length_y; j++) {
            vy.push_back(EmbeddingSelfEnergy(parameters, kx.at(i), ky.at(j)));
        }
        leads.push_back(vy);
    }

    /*
    std::cout << "kx is " << std::endl;
    for (int i = 0; i < parameters.chain_length_x; i++) {
        std::cout << kx.at(i) << ",";
    }
    std::cout << "\n" << "ky is " << std::endl;
    for (int i = 0; i < parameters.chain_length_y; i++) {
        std::cout << ky.at(i) << "," << "\n";
    }
    */
    std::cout << "leads complete" << std::endl;
    get_local_gf(parameters, kx, ky, self_energy_mb_up, leads, gf_local_up, gf_local_down);

    dmft(parameters, parameters.voltage_step, kx, ky, self_energy_mb_up, self_energy_mb_down, 
        gf_local_up, gf_local_down);

    if(parameters.hubbard_interaction == 0 && parameters.chain_length == 1 && parameters.chain_length_x == 1){
        get_analytic_gf_1_site(parameters, gf_local_up);
    }

    std::ofstream myfile2;
    myfile2.open("/home/declan/green_function_code/quantum_transport/textfiles/gf_c++.txt");
    // myfile << parameters.steps << std::endl;
    for(int i = 0; i < parameters.chain_length; i++){  
        for (int r = 0; r < parameters.steps; r++)
        {
            myfile2 << parameters.energy.at(r).real() << "," << gf_local_up.at(r)(i, i).real() << "," << gf_local_up.at(r)(i, i).imag() << "\n";
            // std::cout << leads.self_energy_left.at(r) << "\n";
        }
    }
    myfile2.close();

    return 0;
}