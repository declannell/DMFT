#include "parameters.h"
#include "leads_self_energy.h"
#include "dmft.h"
#include "interacting_gf.h"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <eigen3/Eigen/Dense>
#include <iomanip>


void get_spin_occupation(Parameters &parameters, std::vector<dcomp> &gf_lesser_up,
                        std::vector<dcomp> &gf_lesser_down, double *spin_up, double *spin_down){
    double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
    double result_up = 0.0, result_down = 0.0;
    for(int r = 0; r < parameters.steps; r++){
        result_up = (delta_energy) * gf_lesser_up.at(r).imag() + result_up;
        result_down = (delta_energy) * gf_lesser_down.at(r).imag() + result_down;
    }
    *spin_up = 1.0 / (2.0 * M_PI) * result_up;
    *spin_down = 1.0 / (2.0 * M_PI) * result_down;

   //std::cout << *spin_up << std::endl;
}


void get_difference(Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &old_green_function,
                double &difference){
    difference = -std::numeric_limits<double>::infinity();
    double real_difference, imag_difference;
    for (int r = 0; r < parameters.steps; r++) {
        for(int i = 0; i < parameters.chain_length; i++){
            for(int j = 0; j < parameters.chain_length; j++){
                real_difference = abs(gf_local_up.at(r)(i, j).real() - old_green_function.at(r)(i, j).real());
                imag_difference = abs(gf_local_up.at(r)(i, j).imag() - old_green_function.at(r)(i, j).imag());
                difference = std::max(difference, std::max(real_difference, imag_difference));
                old_green_function.at(r)(i, j) = gf_local_up.at(r)(i, j);
            }
        }
    }
}

void fluctuation_dissipation(Parameters &parameters, const std::vector<dcomp> &green_function, std::vector<dcomp> &lesser_green_function){
    for(int r = 0; r < parameters.steps; r++){
        lesser_green_function.at(r) = - 1.0 * fermi_function(parameters.energy.at(r).real(), parameters) * (
            green_function.at(r) - std::conj(green_function.at(r)));
        //std::cout << lesser_green_function.at(r) << std::endl;
    }
    //std::cout << "\n";

    std::ofstream myfile2;
    myfile2.open("/home/declan/green_function_code/quantum_transport/textfiles/gf_lesser_c++.txt");

    for (int r = 0; r < parameters.steps; r++)
    {
        myfile2 << parameters.energy.at(r).real() << "," << lesser_green_function.at(r).real() << "," << lesser_green_function.at(r).imag() << "\n";
    }
    myfile2.close();    
}


dcomp integrate(Parameters &parameters, std::vector<dcomp> &gf_1, std::vector<dcomp> &gf_2,
            std::vector<dcomp> &gf_3, int r){
    double delta_energy = (parameters.e_upper_bound -
                    parameters.e_lower_bound) / (double)parameters.steps;
    dcomp result = 0;
    for(int i = 0; i < parameters.steps; i++) {
        for(int j = 0; j < parameters.steps; j++){
            if (((i + j - r) >= 0) && ((i + j - r) < parameters.steps)) {
                //this integrates like PHYSICAL REVIEW B 74, 155125 2006
                //I say the green function is zero outside e_lower_bound and e_upper_bound. This means I need the final green function in the integral to be within an energy of e_lower_bound
                //and e_upper_bound. The index of 0 corresponds to e_lower_bound. Hence we need i+J-r>0 but in order to be less an energy of e_upper_bound we need i+j-r<steps. 
                //These conditions enesure the enrgy of the gf3 greens function to be within (e_upper_bound, e_lower_bound)
                result = (delta_energy / (2.0 * parameters.pi)) * (delta_energy / (2.0 * parameters.pi)) * 
                    gf_1.at(i) * gf_2.at(j) * gf_3.at(i + j - r) + result;
            } 
        }
    }
    return result;
}

void self_energy_2nd_order(Parameters &parameters, std::vector<dcomp> &impurity_gf_up, std::vector<dcomp> &impurity_gf_down, 
        std::vector<dcomp> &impurity_gf_up_lesser, std::vector<dcomp> &impurity_gf_down_lesser, std::vector<dcomp> &impurity_self_energy,
        std::vector<dcomp> &impurity_self_energy_lesser_up){
    
    std::vector<dcomp> impurity_gf_down_advanced(parameters.steps), gf_greater_down(parameters.steps);
    
    for(int r = 0; r < parameters.steps; r++){
        impurity_gf_down_advanced.at(r) = std::conj(impurity_gf_down.at(r));
        gf_greater_down.at(r) = parameters.j1 * 2.0 * impurity_gf_down.at(r).imag() + impurity_gf_down_lesser.at(r);
    }   
    /*
    std::cout << "The greater green function is \n";
    for(int r = 0 ; r < parameters.steps; r++){
        std::cout << gf_greater_down.at(r) << std::endl;
    }
        std::cout << "\n";

    std::cout << "The lesser green function is \n";
    for(int r = 0 ; r < parameters.steps; r++){
        std::cout << impurity_gf_down_lesser.at(r) << std::endl;
    }
        std::cout << "\n";
    */
    for(int r = 0; r < parameters.steps; r++){
        impurity_self_energy.at(r) = parameters.hubbard_interaction * parameters.hubbard_interaction * 
            (integrate(parameters, impurity_gf_up, impurity_gf_down,impurity_gf_down_lesser, r));  // line 3

        impurity_self_energy.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction *
            (integrate(parameters, impurity_gf_up, impurity_gf_down_lesser, impurity_gf_down_lesser, r));  // line 2

        impurity_self_energy.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction * 
        (integrate(parameters, impurity_gf_up_lesser, impurity_gf_down, impurity_gf_down_lesser, r));  // line 1

        impurity_self_energy.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction * 
        (integrate(parameters, impurity_gf_up_lesser, impurity_gf_down_lesser, impurity_gf_down_advanced, r));  //line 4
        //there could be a problem here with the constant multiplying sigma lesser
        impurity_self_energy_lesser_up.at(r) = parameters.hubbard_interaction * parameters.hubbard_interaction *
        (integrate(parameters, impurity_gf_up_lesser, impurity_gf_down_lesser, gf_greater_down, r));  

        
        //std::cout << impurity_self_energy.at(r) << impurity_self_energy_lesser_up.at(r) << "\n";
    }
}

void impurity_solver(Parameters &parameters, std::vector<dcomp>  &impurity_gf_up, std::vector<dcomp>  &impurity_gf_down,
    std::vector<dcomp>  &impurity_gf_up_lesser, std::vector<dcomp>  &impurity_gf_down_lesser,
    std::vector<dcomp>  &impurity_self_energy_up, std::vector<dcomp>  &impurity_self_energy_down, 
    std::vector<dcomp>  &impurity_self_energy_lesser_up, std::vector<dcomp>  &impurity_self_energy_lesser_down,
    double *spin_up, double *spin_down){

    get_spin_occupation(parameters, impurity_gf_up_lesser, impurity_gf_down_lesser, spin_up, spin_down);
    std::cout << std::setprecision (15) << "The spin up occupancy is " << *spin_up << "\n";
    std::cout << "The spin down occupancy is " << *spin_down << "\n";
    
    if (parameters.interaction_order == 2){
        self_energy_2nd_order(parameters, impurity_gf_up, impurity_gf_down, impurity_gf_up_lesser,
            impurity_gf_down_lesser, impurity_self_energy_up, impurity_self_energy_lesser_up);
        self_energy_2nd_order(parameters, impurity_gf_down, impurity_gf_up, impurity_gf_down_lesser,
            impurity_gf_up_lesser, impurity_self_energy_down, impurity_self_energy_lesser_down);

        for(int r = 0; r < parameters.steps; r++){
            impurity_self_energy_up.at(r) += parameters.hubbard_interaction * (*spin_down);
            impurity_self_energy_down.at(r) += parameters.hubbard_interaction * (*spin_up);
        }

    }

    if (parameters.interaction_order == 1){
        for(int r = 0; r < parameters.steps; r++) {
            impurity_self_energy_up.at(r) = parameters.hubbard_interaction * (*spin_down);
            impurity_self_energy_down.at(r) = parameters.hubbard_interaction * (*spin_up);
        }
    }
}




void dmft(Parameters &parameters, int voltage_step, std::vector<double> const &kx, std::vector<double> const &ky, 
        std::vector<std::vector<dcomp>> &self_energy_mb_up, std::vector<std::vector<dcomp>> &self_energy_mb_down, 
        std::vector<std::vector<dcomp>> &self_energy_mb_lesser_up, std::vector<std::vector<dcomp>> &self_energy_mb_lesser_down,
        std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &gf_local_down, 
        std::vector<Eigen::MatrixXcd> &gf_local_lesser_up, std::vector<Eigen::MatrixXcd> &gf_local_lesser_down,
        std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<double> &spins_occup){

    double difference = std::numeric_limits<double>::infinity();
    int count = 0;
    std::vector<Eigen::MatrixXcd> old_green_function(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
    while (difference > 0.0001 && count < parameters.self_consistent_steps){
        get_difference(parameters, gf_local_up, old_green_function, difference);
        std::cout << "The difference is " << difference <<". The count is " << count << std::endl;
        if (difference < 0.0000001){
            break;
        }

        for(int i = 0; i < parameters.chain_length; i++) {
            std::vector<dcomp> diag_gf_local_up(parameters.steps), diag_gf_local_down(parameters.steps), 
                diag_gf_local_lesser_up(parameters.steps), diag_gf_local_lesser_down(parameters.steps),
                impurity_self_energy_up(parameters.steps), impurity_self_energy_down(parameters.steps),
                impurity_self_energy_lesser_up(parameters.steps), impurity_self_energy_lesser_down(parameters.steps);

            for(int r = 0; r < parameters.steps; r++){
                diag_gf_local_up.at(r) = gf_local_up.at(r)(i, i);
                diag_gf_local_down.at(r) = gf_local_down.at(r)(i, i);
                diag_gf_local_lesser_up.at(r) = gf_local_lesser_up.at(r)(i, i);
                diag_gf_local_lesser_down.at(r) = gf_local_lesser_down.at(r)(i, i);
            }

            impurity_solver(parameters, diag_gf_local_up, diag_gf_local_down, diag_gf_local_lesser_up, diag_gf_local_lesser_down,
            impurity_self_energy_up, impurity_self_energy_down, impurity_self_energy_lesser_up,
            impurity_self_energy_lesser_down, &spins_occup.at(i), &spins_occup.at(i + parameters.chain_length));

            for(int r = 0; r < parameters.steps; r++){
                self_energy_mb_up.at(i).at(r) = impurity_self_energy_up.at(r);
                self_energy_mb_down.at(i).at(r) = impurity_self_energy_down.at(r);
                self_energy_mb_lesser_up.at(i).at(r) = impurity_self_energy_lesser_up.at(r);
                self_energy_mb_lesser_down.at(i).at(r) = impurity_self_energy_lesser_down.at(r);
            }
        }
        get_local_gf_r_and_lesser(parameters, kx, ky, self_energy_mb_up, self_energy_mb_lesser_up,
            leads, gf_local_up, gf_local_lesser_up, voltage_step);
        get_local_gf_r_and_lesser(parameters, kx, ky, self_energy_mb_down, self_energy_mb_lesser_down,
            leads, gf_local_down, gf_local_lesser_down, voltage_step);

        if(voltage_step == 0){
            std::vector<Eigen::MatrixXcd> gf_local_lesser_up_FD(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
            for(int r = 0; r < parameters.steps; r++){
                for(int i = 0; i < parameters.chain_length; i++){
                    for(int j = 0; j < parameters.chain_length; j++){
                        gf_local_lesser_up_FD.at(r)(i, j) = - 1.0 * fermi_function(parameters.energy.at(r).real(), parameters) *
                            (gf_local_up.at(r)(i, j) - std::conj(gf_local_up.at(r)(j, i)));
                        
                    }
                }
                //std::cout << gf_local_lesser_up_FD.at(r) << std::endl;
            }
            double difference;
            
            get_difference(parameters, gf_local_lesser_up, gf_local_lesser_up_FD, difference);

            std::cout << "The difference between the fD and other is " << difference << std::endl;


            std::ofstream myfile9;
            myfile9.open("/home/declan/green_function_code/quantum_transport/textfiles/gf_lesser_FD_c++.txt");
            // myfile << parameters.steps << std::endl;
            for(int i = 0; i < parameters.chain_length; i++){  
                for (int r = 0; r < parameters.steps; r++)
                {
                    myfile9 << parameters.energy.at(r).real() << "," << gf_local_lesser_up_FD.at(r)(i, i).real() << "," << gf_local_lesser_up_FD.at(r)(i, i).imag() << "\n";
                }
            }
            myfile9.close();
        }

        std::ofstream myfile1;
        myfile1.open("/home/declan/green_function_code/quantum_transport/textfiles/gf_lesser_c++.txt");
        // myfile << parameters.steps << std::endl;
        for(int i = 0; i < parameters.chain_length; i++){  
            for (int r = 0; r < parameters.steps; r++)
            {
                myfile1 << parameters.energy.at(r).real() << "," << gf_local_lesser_up.at(r)(i, i).real() << "," << gf_local_lesser_up.at(r)(i, i).imag() << "\n";
            }
        }
        myfile1.close();



        count++;
    }
}

