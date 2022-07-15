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

Interacting_GF::Interacting_GF(const Parameters &parameters, const std::vector<std::vector<dcomp>> &self_energy_mb, const  std::vector<dcomp> &self_energy_left,
                const std::vector<dcomp> &self_energy_right, const int voltage_step, const  Eigen::MatrixXd &hamiltonian)
{
    this->interacting_gf.resize(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
    get_interacting_gf(parameters, hamiltonian, self_energy_mb, self_energy_left,
                                              self_energy_right, voltage_step);
}

void Interacting_GF::get_interacting_gf(const Parameters &parameters, const Eigen::MatrixXcd& hamiltonian, const std::vector<std::vector<dcomp>> &self_energy_mb, 
                            std::vector<dcomp> const &self_energy_left, std::vector<dcomp> const &self_energy_right, const int voltage_step){
    Eigen::MatrixXcd inverse_gf;
    for(int r = 0; r < parameters.steps; r++){
        inverse_gf = Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length);
        if (parameters.chain_length != 1){
            inverse_gf(0, 0) = parameters.energy.at(r) + parameters.j1 * parameters.delta_gf - hamiltonian(0, 0) - self_energy_mb.at(0).at(r) - self_energy_left.at(r);

            inverse_gf(parameters.chain_length - 1, parameters.chain_length - 1) = parameters.energy.at(r) + parameters.j1 * parameters.delta_gf  - 
                hamiltonian(parameters.chain_length - 1, parameters.chain_length - 1) - self_energy_mb.at(parameters.chain_length - 1).at(r) - self_energy_right.at(r);

        } else if (parameters.chain_length == 1) {
            inverse_gf(0, 0) = parameters.energy.at(r) + parameters.j1 * parameters.delta_gf  - hamiltonian(0, 0) - self_energy_mb.at(0).at(r) - self_energy_left.at(r) - self_energy_right.at(r);
        }

        for(int i = 0; i < parameters.chain_length; i++){
            for(int j = 0; j < parameters.chain_length; j++){
                if (i == j && ((i != 0) && (i != parameters.chain_length - 1))) {
                    //std::cout << i  << j << std::endl;
                    inverse_gf(i, i) = parameters.energy.at(r) + parameters.j1 * parameters.delta_gf  - hamiltonian(i, i) - self_energy_mb.at(i).at(r);
                } else if (i != j) {
                    inverse_gf(i, j) = - hamiltonian(i, j);
                }
            }           
        }
    this->interacting_gf.at(r) = inverse_gf.inverse();
    //std::cout << "The inverse of A is:\n" << interacting_gf.at(r)(0, 0) << std::endl;

    }
}

void get_hamiltonian(Parameters const &parameters, const int voltage_step, const double kx, const double ky, Eigen::MatrixXd &hamiltonian){
    for (int i = 0; i < parameters.num_ins_left - 1; i++){
        hamiltonian(i, i + 1) = parameters.hopping_ins_l;
        hamiltonian(i + 1, i) = parameters.hopping_ins_l;
    }
    //std::cout << "failed here 1 \n"; 
    for (int i = 0; i < parameters.num_cor - 1; i++){
        int j = i + parameters.num_ins_left;
        //std::cout << j << std::endl;
        hamiltonian(j, j + 1) = parameters.hopping_cor;
        hamiltonian(j + 1, j) = parameters.hopping_cor;
    }

    //std::cout << "failed here 2 \n"; 

    for (int i = 0; i < parameters.num_ins_right - 1; i++){
        int j = i + parameters.num_cor + parameters.num_ins_left;
        hamiltonian(j, j + 1) = parameters.hopping_ins_r;
        hamiltonian(j + 1, j) = parameters.hopping_ins_r;
    }

    //std::cout << "failed here 3 \n"; 

    if(parameters.num_ins_left != 0 && parameters.num_cor != 0){
        hamiltonian(parameters.num_ins_left, parameters.num_ins_left - 1) = parameters.hopping_ins_l_cor;
        hamiltonian(parameters.num_ins_left - 1, parameters.num_ins_left) = parameters.hopping_ins_l_cor;
    }

    //std::cout << "failed here 4 \n"; 

    if(parameters.num_ins_right != 0 && parameters.num_cor != 0){
        int i = parameters.num_ins_left + parameters.num_cor;
        hamiltonian(i, i - 1) = parameters.hopping_ins_r_cor;
        hamiltonian(i - 1, i) = parameters.hopping_ins_r_cor;
    }


    std::ofstream potential_file;
    potential_file.open(
        "textfiles/"
        "potential.txt");
    potential_file << -5 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -4 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -3 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -2 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -1 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << 0 << "  " << parameters.voltage_l[voltage_step] <<  "\n";


    double potential_bias = (parameters.voltage_l[voltage_step] -
                             parameters.voltage_r[voltage_step]);

    double voltage_i;

    int num_ins = parameters.num_ins_left + parameters.num_ins_right;
    //std::cout << "failed here 5 \n"; 
    for (int i = 0; i < parameters.num_ins_left; i++){
        voltage_i = parameters.voltage_l[voltage_step] - (double)(i + 1) / (double)(num_ins + 1.0) * potential_bias;
        potential_file << i + 1 << "  " << voltage_i <<  "\n";        
        hamiltonian(i, i) = parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l + 2 * parameters.hopping_x * cos(kx) + 2 * parameters.hopping_y * 
                cos(ky) + voltage_i;
    }

    if(parameters.num_ins_left == 0 && parameters.num_ins_right != 0){
        voltage_i = parameters.voltage_l[voltage_step];
    } else if (parameters.num_ins_right == 0 && parameters.num_ins_left != 0){
        voltage_i = parameters.voltage_r[voltage_step];
    } else if (parameters.num_ins_right == 0 && parameters.num_ins_left == 0){
        voltage_i = (parameters.voltage_l[voltage_step] + parameters.voltage_r[voltage_step]) * 0.5;
    } else { 
        voltage_i = parameters.voltage_l[voltage_step] - (double)(parameters.num_ins_left + 0.5) / (double)(num_ins + 1.0) * potential_bias;
        //the voltage is constant in the correlated metal
    }

    //std::cout << "The voltage on the correlated atom is " << voltage_i << std::endl;

    for (int i = 0; i < parameters.num_cor; i++){
        int j = i + parameters.num_ins_left;
        potential_file << j + 1 << "  " << voltage_i <<  "\n";          
        hamiltonian(j, j) = parameters.onsite_cor + 2 * parameters.hopping_x * cos(kx) + 2 * parameters.hopping_y * 
                cos(ky) + voltage_i;
    }

    for (int i = 0; i < parameters.num_ins_right; i++){
        int j = i + parameters.num_cor + parameters.num_ins_left;
        voltage_i = parameters.voltage_l[voltage_step] - (double)(i + parameters.num_ins_left + 1) / (double)(num_ins + 1.0) * potential_bias;
        potential_file << j + 1 << "  " << voltage_i <<  "\n";  
        hamiltonian(j, j) = parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r + 2 * parameters.hopping_x * cos(kx) + 2 * parameters.hopping_y * 
                cos(ky) + voltage_i;
    }

    //std::cout << "The hamiltonian is " <<  std::endl;
    //std::cout << hamiltonian << std::endl;
    //std::cout << std::endl;

    potential_file << parameters.chain_length + 1  << "  " << parameters.voltage_r[voltage_step] <<  "\n";
    potential_file << parameters.chain_length + 2  << "  " << parameters.voltage_r[voltage_step] <<  "\n";
    potential_file << parameters.chain_length + 3  << "  " << parameters.voltage_r[voltage_step] <<  "\n";
    potential_file << parameters.chain_length + 4  << "  " << parameters.voltage_r[voltage_step] <<  "\n";
    potential_file << parameters.chain_length + 5  << "  " << parameters.voltage_r[voltage_step] <<  "\n";
    potential_file << parameters.chain_length + 6  << "  " << parameters.voltage_r[voltage_step] <<  "\n";
    potential_file.close();
}

void get_analytic_gf_1_site(Parameters &parameters, std::vector<Eigen::MatrixXcd> &green_function, int voltage_step){
    std::vector<dcomp> analytic_gf(parameters.steps);

    EmbeddingSelfEnergy leads(parameters, M_PI / 2.0, M_PI / 2.0, voltage_step);
    double difference = -std::numeric_limits<double>::infinity();
    
    for(int r = 0; r < parameters.steps; r++){
        dcomp x = parameters.energy.at(r) + parameters.j1 * parameters.delta_gf - parameters.onsite_cor - 2.0 * parameters.hopping_x * cos(M_PI / 2.0) 
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
    myfile.open("textfiles/gf_c++_analytic.txt");
    // myfile << parameters.steps << std::endl;
    for (int r = 0; r < parameters.steps; r++)
    {
        myfile << analytic_gf.at(r).real() << "," << analytic_gf.at(r).imag() << "\n";
        // std::cout << leads.self_energy_left.at(r) << "\n";
    }

    myfile.close();
}

void get_local_gf(Parameters &parameters, std::vector<std::vector<dcomp>> &self_energy_mb, 
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local, int voltage_step, const std::vector<std::vector<Eigen::MatrixXd>> &hamiltonian){

    int n_x, n_y;
    if (parameters.leads_3d == false){
        n_x =  parameters.num_kx_points; //number of k points to take in x direction
        n_y =  parameters.num_ky_points; //number of k points to take in y direction
    } else {
        n_x = 1;
        n_y = 1;
    }
    double num_k_points = n_x * n_y;

    for(int kx_i = 0; kx_i < n_x; kx_i++) {
        for(int ky_i = 0; ky_i < n_y; ky_i++) {

            Interacting_GF gf_interacting(parameters, self_energy_mb,
                leads.at(kx_i).at(ky_i).self_energy_left,
                leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, hamiltonian.at(kx_i).at(ky_i));

            for(int r = 0; r < parameters.steps; r++){
                gf_local.at(r) = (Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
                for(int i = 0; i < parameters.chain_length; i++){
                    for(int j = 0; j < parameters.chain_length; j++){
                        gf_local.at(r)(i, j) += gf_interacting.interacting_gf.at(r)(i, j) / num_k_points;
                    }
                }
            }
        }
    }
}

void get_advance_gf(const Parameters &parameters, const Eigen::MatrixXcd &gf_retarded, Eigen::MatrixXcd &gf_advanced){
    for(int i = 0; i < parameters.chain_length; i++){
        for(int j = 0; j < parameters.chain_length; j++){
            gf_advanced(i, j) = std::conj(gf_retarded(j, i));
        }
    } 
}



void get_gf_lesser_non_eq(const Parameters &parameters, const std::vector<Eigen::MatrixXcd> &gf_retarded, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const std::vector<dcomp> &self_energy_left,
    const std::vector<dcomp> &self_energy_right, std::vector<Eigen::MatrixXcd> &gf_lesser, int voltage_step){


    for(int r = 0; r < parameters.steps; r++) {
        gf_lesser.at(r) = (Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
    }

    Eigen::MatrixXcd delta_term = Eigen::MatrixXd::Zero(parameters.chain_length, parameters.chain_length);
        //std::cout << "The lesser green function is" << "\n";
    for(int r = 0; r < parameters.steps; r++) {   
    
        delta_term = 2.0 * parameters.j1 * parameters.delta_gf * fermi_function(parameters.energy.at(r), parameters) 
            * gf_retarded.at(r) * gf_retarded.at(r).adjoint();

        for(int i = 0; i < parameters.chain_length; i++ ) {
            for(int j = 0; j < parameters.chain_length; j++) {  
                for(int k = 0; k < parameters.chain_length; k++){
                    gf_lesser.at(r)(i, j) += gf_retarded.at(r)(i, k) * (self_energy_mb_lesser.at(k).at(r)) * std::conj(gf_retarded.at(r)(j, k));
                    if (k == 0){
                        gf_lesser.at(r)(i, j) += gf_retarded.at(r)(i, k) * (- 2.0 * parameters.j1) * fermi_function(parameters.energy.at(r) - parameters.voltage_l.at(voltage_step), parameters) * 
                            (self_energy_left.at(r)).imag() * std::conj(gf_retarded.at(r)(j, k));
                    }
                    if (k == parameters.chain_length - 1){
                        //std::cout <<  parameters.voltage_r.at(voltage_step) << std::endl;
                        gf_lesser.at(r)(i, j) += gf_retarded.at(r)(i, k) * (- 2.0 * parameters.j1)  * fermi_function(parameters.energy.at(r) - parameters.voltage_r.at(voltage_step), parameters) * 
                            (self_energy_right.at(r)).imag() * std::conj(gf_retarded.at(r)(j, k));
                        
                        
                    }
                } 
            }
        }
        gf_lesser.at(r) = gf_lesser.at(r) + delta_term;
    }
}

void get_gf_lesser_fd(const Parameters &parameters, const std::vector<Eigen::MatrixXcd> &gf_retarded, std::vector<Eigen::MatrixXcd> &gf_lesser){
    for(int r = 0; r < parameters.steps; r++){
        for(int i = 0; i < parameters.chain_length; i++){
            for(int j = 0; j < parameters.chain_length; j++){
                gf_lesser.at(r)(i, j) = - 1.0 * fermi_function(parameters.energy.at(r), parameters) *
                    (gf_retarded.at(r)(i, j) - std::conj(gf_retarded.at(r)(j, i)));
            }
        }
    }
}

void get_local_gf_r_and_lesser(const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local, 
    std::vector<Eigen::MatrixXcd> &gf_local_lesser, const int voltage_step, const std::vector<std::vector<Eigen::MatrixXd>> &hamiltonian){


    for(int r = 0; r < parameters.steps; r++){
        gf_local.at(r) = (Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
        gf_local_lesser.at(r) = (Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
    }

    int n_x, n_y;

    if (parameters.leads_3d == false){
        n_x =  parameters.num_kx_points; //number of k points to take in x direction
        n_y =  parameters.num_ky_points; //number of k points to take in y direction
    } else {
        n_x = 1;
        n_y = 1;
    }

    std::vector<Eigen::MatrixXcd> gf_lesser(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length)); 
    double num_k_points = n_x * n_y;
    for(int kx_i = 0; kx_i < n_x; kx_i++) {
        for(int ky_i = 0; ky_i < n_y; ky_i++) {
            Interacting_GF gf_interacting(parameters, self_energy_mb,
                leads.at(kx_i).at(ky_i).self_energy_left,
                leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, hamiltonian.at(kx_i).at(ky_i));


            get_gf_lesser_non_eq(parameters, gf_interacting.interacting_gf, 
                self_energy_mb_lesser, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
                gf_lesser, voltage_step);



            for(int r = 0; r < parameters.steps; r++){
                for(int i = 0; i < parameters.chain_length; i++){
                    for(int j = 0; j < parameters.chain_length; j++){
                        gf_local.at(r)(i, j) += gf_interacting.interacting_gf.at(r)(i, j) / num_k_points;
                        gf_local_lesser.at(r)(i, j) += gf_lesser.at(r)(i, j) / num_k_points;
                    }
                }
            }
        }
    }
}

