#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include "parameters.h"
#include "leads_self_energy.h"
#include <eigen3/Eigen/Dense>
// void function(vector<vector<vector<double>>> &green_function )

// this is a fake make file
// g++ -g -O -c parameters.cpp
// g++ -g -Wall  .\leads_self_energy.cpp -lm parameters.o
// g++ -g -Wall  main.cpp -lm parameters.o
double EmbeddingSelfEnergy::kx() const { return kx_value; }
double EmbeddingSelfEnergy::ky() const { return ky_value; }

EmbeddingSelfEnergy::EmbeddingSelfEnergy(const Parameters &parameters, double kx, double ky, int voltage_step) : kx_value(kx), ky_value(ky) // type is implied, it knows this is a constructor
{
    this->self_energy_left.resize(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
    this->self_energy_right.resize(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length)); 

    if(parameters.wbl_approx == true){
        for (int r = 0; r < parameters.steps_myid; r++){
            for (int i = 0; i < 4; i++){
                self_energy_left.at(r)(i * parameters.chain_length, i * parameters.chain_length) = parameters.j1 * parameters.gamma * 0.5;
                self_energy_right.at(r)((i + 1) * parameters.chain_length - 1, (i + 1) * parameters.chain_length - 1) = parameters.j1 * parameters.gamma * 0.5;               
            }
        }
    } else { //this calculate the self energy via the method in sanchez paper. This is not tested.
        Eigen::Matrix4cd  hamiltonian(4, 4);

        get_hamiltonian_for_leads(parameters, hamiltonian);
        MatrixVectorType transfer_matrix_l(parameters.steps_myid, MatrixType::Zero(4, 4));
        MatrixVectorType transfer_matrix_r(parameters.steps_myid, MatrixType::Zero(4, 4));
        get_transfer_matrix(parameters, transfer_matrix_l, transfer_matrix_r, hamiltonian, voltage_step);
        get_self_energy(parameters, transfer_matrix_l, transfer_matrix_r, hamiltonian, voltage_step);

        transfer_matrix_l.clear();
        transfer_matrix_l.shrink_to_fit();
        transfer_matrix_r.clear();
        transfer_matrix_r.shrink_to_fit();
    }
}

void EmbeddingSelfEnergy::get_hamiltonian_for_leads(const Parameters &parameters, Eigen::Matrix4cd &hamiltonian) {
    // Set onsite energy for each site
    for (int i = 0; i < 4; i++) {
        hamiltonian(i, i) = parameters.onsite_l;
    }
    // Calculate multipliers for hopping terms
    dcomp multiple_upper_x = 1.0, multiple_upper_y = 1.0, multiple_lower_x = 1.0, multiple_lower_y = 1.0;

    if (parameters.num_ky_points != 1) {
        multiple_lower_y += exp(-parameters.j1 * ky());
        multiple_upper_y += exp(parameters.j1 * ky());
    }

    if (parameters.num_kx_points != 1) {
        multiple_lower_x += exp(-parameters.j1 * kx());
        multiple_upper_x += exp(parameters.j1 * kx());
    }

    // Populate the Hamiltonian matrix with hopping terms
    hamiltonian(0, 1) = parameters.hopping_lx * multiple_upper_x;
    hamiltonian(0, 2) = parameters.hopping_ly * multiple_upper_y;
    hamiltonian(1, 3) = parameters.hopping_ly * multiple_upper_y;
    hamiltonian(2, 3) = parameters.hopping_lx * multiple_upper_x;

    hamiltonian(1, 0) = parameters.hopping_lx * multiple_lower_x;
    hamiltonian(2, 0) = parameters.hopping_ly * multiple_lower_y;
    hamiltonian(3, 1) = parameters.hopping_ly * multiple_lower_y;
    hamiltonian(3, 2) = parameters.hopping_lx * multiple_lower_x;
}



void EmbeddingSelfEnergy::get_transfer_matrix(const Parameters &parameters, MatrixVectorType &transfer_matrix_l, 
    MatrixVectorType &transfer_matrix_r, Eigen::Matrix4cd &hamiltonian, int voltage_step)
{   //this followsM P Lopez Sancho et al 1984 J. Phys. F: Met. Phys. 14 1205. For me t_i = \tilde{t}_i
    MatrixVectorType t_i_left(parameters.steps_myid, MatrixType::Zero(4, 4));
    MatrixVectorType t_i_right(parameters.steps_myid, MatrixType::Zero(4, 4));
    MatrixVectorType t_product_l(parameters.steps_myid, MatrixType::Zero(4, 4));
    MatrixVectorType t_product_r(parameters.steps_myid, MatrixType::Zero(4, 4));
    Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();

    for (int r = 0; r < parameters.steps_myid; r++) {
        int y = r + parameters.start.at(parameters.myid);
        Eigen::Matrix4cd inverse_l = (parameters.energy.at(y) + parameters.j1 * parameters.delta_leads 
            - parameters.voltage_l[voltage_step]) * identity - hamiltonian;
        
        Eigen::Matrix4cd inverse_r = (parameters.energy.at(y) + parameters.j1 * parameters.delta_leads 
            - parameters.voltage_r[voltage_step]) * identity - hamiltonian;

        t_i_left.at(r) = parameters.hopping_lz * inverse_l.inverse();
        t_i_right.at(r) = parameters.hopping_rz * inverse_r.inverse();

        t_product_l.at(r) = t_i_left.at(r);
        t_product_r.at(r) = t_i_right.at(r);
        transfer_matrix_l.at(r) = t_i_left.at(r);
        transfer_matrix_r.at(r) = t_i_right.at(r);

        //if (y == 0) std::cout << inverse_l - inverse_r << std::endl;
    }

    

    MatrixVectorType old_transfer(parameters.steps, MatrixType::Zero(4, 4));

    double difference, real_difference, imag_difference;
    int count = 0;
    do
    {
        difference = -std::numeric_limits<double>::infinity();
        for (int r = 0; r < parameters.steps_myid; r++)
        {
            Eigen::Matrix4cd t_l_squared = t_i_left.at(r) * t_i_left.at(r);
            Eigen::Matrix4cd t_r_squared = t_i_right.at(r) * t_i_right.at(r); 

            t_i_left.at(r) = (identity - 2.0 * t_l_squared).inverse() * t_l_squared;
            t_i_right.at(r) = (identity - 2.0 * t_r_squared).inverse() * t_r_squared;

            t_product_l.at(r) = t_product_l.at(r) * t_i_left.at(r);
            t_product_r.at(r) = t_product_r.at(r) * t_i_right.at(r);

            transfer_matrix_l.at(r) = transfer_matrix_l.at(r) + t_product_l.at(r);
            transfer_matrix_r.at(r) = transfer_matrix_r.at(r) + t_product_r.at(r);

            for(int i = 0; i < 4; i++){
                for(int j = 0; j < 4; j++){
                    real_difference = abs(transfer_matrix_l.at(r)(i, j).real() - old_transfer.at(r)(i, j).real());
                    imag_difference = abs(transfer_matrix_l.at(r)(i, j).imag() - old_transfer.at(r)(i, j).imag());
                    difference = std::max(difference, std::max(real_difference, imag_difference));
                }
            }
            old_transfer.at(r) = transfer_matrix_l.at(r);
        }
        count++;

        if (parameters.myid == 0) std::cout << "The leads difference is " << difference << ". The count is " << count << std::endl;

    } while (difference > parameters.convergence_leads && count < parameters.self_consistent_steps_leads);
}

void EmbeddingSelfEnergy::get_self_energy(const Parameters &parameters, MatrixVectorType &transfer_matrix_l, 
    MatrixVectorType &transfer_matrix_r, Eigen::Matrix4cd &hamiltonian, int voltage_step)
{
    Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
    
    MatrixVectorType surface_gf_l(parameters.steps_myid, MatrixType::Zero(4, 4));
    MatrixVectorType surface_gf_r(parameters.steps_myid, MatrixType::Zero(4, 4));

    MatrixType sr_coupling_l = MatrixType::Zero(4 * parameters.chain_length, 4);
    MatrixType sr_coupling_r = MatrixType::Zero(4 * parameters.chain_length, 4);

    for (int i = 0; i < 4; i++) sr_coupling_l(i * parameters.chain_length, i) = parameters.hopping_lc;
    for (int i = 0; i < 4; i++) sr_coupling_r((i + 1) * parameters.chain_length - 1, i) = parameters.hopping_lc;

    for (int r = 0; r < parameters.steps_myid; r++)
    {
        int y = r + parameters.start.at(parameters.myid);
        surface_gf_l.at(r) = ((parameters.j1 * parameters.delta_leads + parameters.energy.at(y) - parameters.voltage_l[voltage_step]) * identity - hamiltonian 
            - parameters.hopping_lz * transfer_matrix_l.at(r)).inverse(); 

        surface_gf_r.at(r) = ((parameters.j1 * parameters.delta_leads + parameters.energy.at(y) - parameters.voltage_r[voltage_step]) * identity - hamiltonian 
            - parameters.hopping_rz * transfer_matrix_r.at(r)).inverse(); 

        this->self_energy_left.at(r) = sr_coupling_l * surface_gf_l.at(r) * (sr_coupling_l.transpose());
        this->self_energy_right.at(r) = sr_coupling_r * surface_gf_r.at(r) * (sr_coupling_r.transpose());
    }
}

/*
template <typename T>
int sgn(T val)
{
    T zero_value = T(0);
    if (val == zero_value) {
        return 0;
    } else if (val > zero_value) {
        return 1;
    } else {
        return -1;
    }
}

std::vector<dcomp> analytic_self_energy(const Parameters &parameters, int voltage_step)
{

    std::vector<dcomp> analytic_se(parameters.steps);
    for (int r = 0; r < parameters.steps; r++)
    {
        double x = (parameters.energy.at(r) - parameters.onsite_l - parameters.voltage_l[voltage_step]) / (2.0 * parameters.hopping_lz);

        analytic_se.at(r) = parameters.hopping_lc * parameters.hopping_lc * (1.0 / abs(parameters.hopping_lz)) * (x + + parameters.j1 * parameters.delta_leads);
        if (abs(x) > 1.0)
        {
            analytic_se.at(r) = analytic_se.at(r).real() -parameters.hopping_lc * parameters.hopping_lc * (1.0 / abs(parameters.hopping_lz)) * (sgn(x) * sqrt(abs(x) * abs(x) - 1.0));
        }
        else if (abs(x) < 1.0)
        {
            analytic_se.at(r) = analytic_se.at(r).real() - parameters.j1 * parameters.hopping_lc * parameters.hopping_lc * abs((1.0 / abs(parameters.hopping_lz))) * (sqrt(1.0 - abs(x) * abs(x)));
        }
    }
    return analytic_se;
}

void run(const Parameters &parameters)
{
    EmbeddingSelfEnergy leads(parameters, M_PI / 2.0, M_PI / 2.0, parameters.voltage_step);

    std::ofstream myfile;
    myfile.open("textfiles/self_energy_lead_real.txt");
    // myfile << parameters.steps << std::endl;
    for (int r = 0; r < parameters.steps; r++)
    {
        myfile << parameters.energy.at(r) << "," << leads.self_energy_left.at(r).real() << "," << leads.self_energy_left.at(r).imag() << "," << leads.self_energy_right.at(r).real() << "," << leads.self_energy_right.at(r).imag() << ","
               << "\n";
        // std::cout << leads.self_energy_left.at(r) << "\n";
    }

    myfile.close();
}

void get_k_averaged_embedding_self_energy(const Parameters parameters, std::vector<std::vector<EmbeddingSelfEnergy>> &leads){
    int num_k = parameters.num_kx_points * parameters.num_ky_points;
    std::vector<dcomp> k_averaged_self_energy_left(parameters.steps), k_averaged_self_energy_right(parameters.steps);
    for( int kx_i = 0; kx_i < parameters.num_kx_points; kx_i++) {
        for( int ky_i = 0; ky_i < parameters.num_ky_points; ky_i++) {
            for(int r = 0; r < parameters.steps; r++) {
                k_averaged_self_energy_left.at(r) += leads.at(kx_i).at(ky_i).self_energy_left.at(r) / (double)num_k;
                k_averaged_self_energy_right.at(r) += leads.at(kx_i).at(ky_i).self_energy_right.at(r) / (double)num_k;
            }
        }
    }

    std::ofstream embedding_se_file;
    embedding_se_file.open(
        "textfiles/"
        "embedding_self_energy.txt");
    // myfile << parameters.steps << std::endl;

	for (int r = 0; r < parameters.steps; r++) {
		embedding_se_file << parameters.energy.at(r) << "  "
				<< k_averaged_self_energy_left.at(r).real() << "  "
				<< k_averaged_self_energy_left.at(r).imag() << "  "
				<< k_averaged_self_energy_right.at(r).real() << "  " 
                << k_averaged_self_energy_right.at(r).imag() <<"\n";
	}
	embedding_se_file.close();
}



*/