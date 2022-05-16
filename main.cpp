#include "parameters.h"
#include "leads_self_energy.h"
#include <iostream>
#include <vector>
#include <x86_64-linux-gnu/mpich/mpi.h>
#include <eigen3/Eigen/Dense>
#include "dmft.h"
#include "interacting_gf.h"
#include "transport.h"


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

    std::vector<dcomp> current_up(parameters.NIV_points, 0);
    std::vector<dcomp> current_down(parameters.NIV_points, 0);

    std::vector<Eigen::MatrixXcd> gf_local_up(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
    std::vector<Eigen::MatrixXcd> gf_local_down(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
    std::vector<std::vector<dcomp>> self_energy_mb_up(parameters.chain_length, std::vector<dcomp> (parameters.steps));
    std::vector<std::vector<dcomp>> self_energy_mb_down(parameters.chain_length, std::vector<dcomp> (parameters.steps));

    for( int m = 0; m < parameters.NIV_points; m++){
        std::cout << "The voltage difference is " <<  parameters.voltage_l[m] - parameters.voltage_r[m] << std::endl;

        std::vector<std::vector<EmbeddingSelfEnergy>> leads;
        for (int i = 0; i < parameters.chain_length_x; i++)
        {
            std::vector<EmbeddingSelfEnergy> vy;
            for (int j = 0; j < parameters.chain_length_y; j++) {
                vy.push_back(EmbeddingSelfEnergy(parameters, kx.at(i), ky.at(j), m));
            }
            leads.push_back(vy);
        }

        std::cout << "leads complete" << std::endl;
        get_local_gf(parameters, kx, ky, self_energy_mb_up, leads, gf_local_up, gf_local_down, m);
        std::cout << "got local green function" << std::endl;

        std::vector<double> spins_occup(2 * parameters.chain_length);

        dmft(parameters, m, kx, ky, self_energy_mb_up, self_energy_mb_down, 
            gf_local_up, gf_local_down, leads, spins_occup);
        std::cout << "got self energy" << std::endl;

        if(parameters.hubbard_interaction == 0 && parameters.chain_length == 1 && parameters.chain_length_x == 1 && m ==0){
            get_analytic_gf_1_site(parameters, gf_local_up, m);
        }

        std::vector<dcomp> transmission_up(parameters.steps, 0);
        std::vector<dcomp> transmission_down(parameters.steps, 0);
        if (parameters.hubbard_interaction == 0) {
            get_transmission(parameters, kx, ky, self_energy_mb_up, 
                leads, transmission_up, transmission_down, m);
            get_landauer_buttiker_current(parameters, transmission_up, transmission_down, 
                &current_up.at(m), &current_down.at(m), m);
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

        std::ofstream myfile3;
        myfile3.open("/home/declan/green_function_code/quantum_transport/textfiles/c++_tranmission.txt");
        // myfile << parameters.steps << std::endl;
        for(int r = 0; r < parameters.steps; r++){  
            myfile3 << parameters.energy.at(r).real() << "," << transmission_up.at(r).real() << "," << transmission_up.at(r).imag() << "," << transmission_down.at(r).real() << "\n";
        }
        myfile2.close();
    }

    std::ofstream myfile4;
    myfile4.open("/home/declan/green_function_code/quantum_transport/textfiles/c++_current.txt");
    // myfile << parameters.steps << std::endl;
    for(int m = 0; m < parameters.NIV_points; m++){  
        myfile4 << parameters.voltage_l[m] - parameters.voltage_r[m] << "," << current_up.at(m).real() <<  "," << current_down.at(m).real() << "\n";
    }
    myfile4.close();


    return 0;
}