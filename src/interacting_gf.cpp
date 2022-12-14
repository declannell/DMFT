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

Interacting_GF::Interacting_GF(const Parameters &parameters, const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<Eigen::MatrixXcd> 
    &self_energy_left, const std::vector<Eigen::MatrixXcd> &self_energy_right, const int voltage_step, const  Eigen::MatrixXcd &hamiltonian)
{
    this->interacting_gf.resize(parameters.steps_myid, Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
    get_interacting_gf(parameters, hamiltonian, self_energy_mb, self_energy_left, self_energy_right, voltage_step);
}

void Interacting_GF::get_interacting_gf(const Parameters &parameters, const Eigen::MatrixXcd& hamiltonian, const std::vector<std::vector<dcomp>> &self_energy_mb, 
        const std::vector<Eigen::MatrixXcd> &self_energy_left, const std::vector<Eigen::MatrixXcd> &self_energy_right, const int voltage_step){
    Eigen::MatrixXcd inverse_gf;

    //std::ostringstream oss1gf;
    //oss1gf << "textfiles/" << "dos_non_int.txt";
    //std::string var1 = oss1gf.str();
    //std::ofstream dos_file_non_int;
    //dos_file_non_int.open(var1);
    // myfile << parameters.steps << std::endl;

    for(int r = 0; r < parameters.steps_myid; r++){
        inverse_gf = Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length);

        for (int i = 0; i < 2 *parameters.chain_length; i++){
            for (int j = 0; j < 2 * parameters.chain_length; j++){
                if(i == j){
                    inverse_gf(i, j) = parameters.energy.at(r + parameters.start.at(parameters.myid)) + parameters.j1 * parameters.delta_gf - hamiltonian(i, j) - self_energy_mb.at(i).at(r);
                } else {
                    inverse_gf(i, j) = - hamiltonian(i, j);
                }
            }
        }

        inverse_gf(0, 0) -= self_energy_left.at(r)(0, 0);
        inverse_gf(parameters.chain_length, parameters.chain_length) -= self_energy_left.at(r)(1, 1);
        inverse_gf(parameters.chain_length - 1, parameters.chain_length - 1) -= self_energy_right.at(r)(0, 0);
        inverse_gf(2 * parameters.chain_length - 1, 2 * parameters.chain_length - 1) -= self_energy_right.at(r)(1, 1);

        if(parameters.wbl_approx != true){
            inverse_gf(0, parameters.chain_length) -= self_energy_left.at(r)(0, 1);
            inverse_gf(parameters.chain_length, 0) -= self_energy_left.at(r)(1, 0);
            inverse_gf(parameters.chain_length - 1, 2 * parameters.chain_length - 1) -= self_energy_right.at(r)(0, 1);
            inverse_gf(2 * parameters.chain_length - 1, parameters.chain_length - 1) -= self_energy_right.at(r)(1, 0);
        }
        


        this->interacting_gf.at(r) = inverse_gf.inverse();
    }
    //dos_file_non_int.close();
}

void get_hamiltonian(Parameters const &parameters, const int voltage_step, const double kx, const double ky, Eigen::MatrixXcd &hamiltonian){
        
    std::ofstream potential_file;
	std::ostringstream ossgf;
	ossgf << voltage_step << ".potential.txt";
	std::string var = ossgf.str();
    potential_file.open(var);
    potential_file << -5 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -4 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -3 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -2 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -1 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << 0 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    double potential_bias = (parameters.voltage_l[voltage_step] -
                            parameters.voltage_r[voltage_step]);
    double voltage_i;

    //the matrix is 2 * chain_length x 2 * chain_length in size. The first block (chain_length x chain_length) is the first layer in the unit cell.
    //The second block (chain_length x chain_length) is the second layer in the unit cell. The offdiagonal blocks are the matrix elements between the two layers.


    for (int i = 0; i < parameters.chain_length - 1; i++){
        //hopping in the first block of the hamiltonian
        if(parameters.atom_type.at(i) == 0 && parameters.atom_type.at(i + 1) == 0){
            hamiltonian(i, i + 1) = parameters.hopping_ins_l;
            hamiltonian(i + 1, i) = parameters.hopping_ins_l;
            //this is for the second layer
            hamiltonian(i + parameters.chain_length, i + parameters.chain_length + 1) = parameters.hopping_ins_l;
            hamiltonian(i + 1 + parameters.chain_length, i + parameters.chain_length) = parameters.hopping_ins_l;

        } else if (parameters.atom_type.at(i) == 1 && parameters.atom_type.at(i + 1) == 1){
            hamiltonian(i, i + 1) = parameters.hopping_cor;
            hamiltonian(i + 1, i) = parameters.hopping_cor; 

            hamiltonian(i + parameters.chain_length, i + parameters.chain_length + 1) = parameters.hopping_cor;
            hamiltonian(i + 1 + parameters.chain_length, i + parameters.chain_length) = parameters.hopping_cor;          
        } else {
            hamiltonian(i, i + 1) = parameters.hopping_ins_l_cor;
            hamiltonian(i + 1, i) = parameters.hopping_ins_l_cor;

            hamiltonian(i + parameters.chain_length, i + parameters.chain_length + 1) = parameters.hopping_ins_l_cor;
            hamiltonian(i + parameters.chain_length + 1, i + parameters.chain_length) = parameters.hopping_ins_l_cor;
        }
        //hopping in the second layer of the hamiltonian
    }    
//    if (parameters.myid == 0) {
        //std::cout << "initialised the hoppings for first layer \n";
    //}

    //we now sort out the hopping on the off diagonal blocks. This doesn't work if I want the hopping of the left and right blocks to be different.
    dcomp multiple_upper = 1.0, multiple_lower = 1.0; //this is the multiple that goes in front of the term for the off diagonal blocks of the green function. 
    
    if (parameters.num_ky_points != 1) {
        multiple_upper += exp(parameters.j1 * ky);
        multiple_lower += exp(-parameters.j1 * ky);
    }
    
    for (int i = 0; i < parameters.chain_length; i++){
        //hopping in the first block of the hamiltonian
        if(parameters.atom_type.at(i) == 0 && parameters.atom_type.at(i + 1) == 0){
            //this is the matrix elements between the two layers
            hamiltonian(i + parameters.chain_length, i) = parameters.hopping_ins_l * multiple_upper;
            hamiltonian(i, i + parameters.chain_length) = parameters.hopping_ins_l * multiple_lower;

        } else if (parameters.atom_type.at(i) == 1 && parameters.atom_type.at(i + 1) == 1){
            hamiltonian(i + parameters.chain_length, i) = parameters.hopping_cor * multiple_upper;
            hamiltonian(i, i + parameters.chain_length) = parameters.hopping_cor * multiple_lower;           
        } else {
            hamiltonian(i + parameters.chain_length, i) = parameters.hopping_ins_l * multiple_upper;              
            hamiltonian(i, i + parameters.chain_length) = parameters.hopping_ins_l * multiple_lower;
        }
        //hopping in the second layer of the hamiltonian
    }    

    //if (parameters.myid == 0) {
    //    std::cout << "initialised the hoppings for second layer \n";
    //}
    if (parameters.ins_metal_ins == true){
        int num_ins = parameters.num_ins_left + parameters.num_ins_right;      
        double delta_v = potential_bias / (double)(num_ins + 1.0);

        //std::cout << "failed here 5 \n"; 
        for (int i = 0; i < parameters.num_ins_left; i++){
            voltage_i = parameters.voltage_l[voltage_step] - (double)(i + 1) * delta_v;
            potential_file << i + 1 << "  " << voltage_i <<  "\n";  
            hamiltonian(i, i) = parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l + voltage_i + 2 * parameters.hopping_ins_l * cos(kx);
            //This is the second layer of the system.
            hamiltonian(i + parameters.chain_length, i + parameters.chain_length) = 
              -1 * (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l) + voltage_i + 2 * parameters.hopping_ins_l * cos(kx);
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
            hamiltonian(j, j) = parameters.onsite_cor + voltage_i + 2 * parameters.hopping_cor * cos(kx);
            hamiltonian(j + parameters.chain_length, j + parameters.chain_length) = parameters.onsite_cor + voltage_i + 2 * parameters.hopping_cor * cos(kx);
        }

        for (int i = 0; i < parameters.num_ins_right; i++){
            int j = i + parameters.num_cor + parameters.num_ins_left;
            voltage_i = parameters.voltage_l[voltage_step] - (double)(i + parameters.num_ins_left + 1) / (double)(num_ins + 1.0) * potential_bias;
            potential_file << j + 1 << "  " << voltage_i <<  "\n";  
            hamiltonian(j, j) = parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r + voltage_i + 2 * parameters.hopping_ins_l * cos(kx);
            hamiltonian(j + parameters.chain_length, j + parameters.chain_length) = -1 * (parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r)
                + 2 * parameters.hopping_ins_l * cos(kx) + voltage_i;
        }

    } else { //this is the metal/ins/metal structure

        int num_ins = parameters.num_ins_left;
        
        double delta_v = potential_bias / (double)(num_ins + 1.0);
        //std::cout << delta_v << std::endl;
        //std::cout << "failed here 5 \n"; 
        voltage_i = parameters.voltage_l[voltage_step];
        for (int i = 0; i < parameters.num_cor; i++){
            potential_file << i + 1 << "  " << voltage_i <<  "\n";  
            hamiltonian(i, i) = parameters.onsite_cor + voltage_i + 2 * parameters.hopping_cor * cos(kx);
            hamiltonian(i + parameters.chain_length, i + parameters.chain_length) = parameters.onsite_cor + voltage_i + 2 * parameters.hopping_cor * cos(kx);
        }

        //std::cout << "The voltage on the correlated atom is " << voltage_i << std::endl;

        for (int i = 0; i < parameters.num_ins_left; i++){
            int j = i + parameters.num_cor;
            voltage_i = parameters.voltage_l[voltage_step] - (double)(i + 1) * delta_v;
            //std::cout << voltage_i << std::endl;
            potential_file << j + 1 << "  " << voltage_i <<  "\n";          
            hamiltonian(j, j) = parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l + 2 * parameters.hopping_lx * cos(kx) + voltage_i 
                + 2 * parameters.hopping_ins_l * cos(kx);
            hamiltonian(j + parameters.chain_length, j + parameters.chain_length) = - (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l) + voltage_i 
                + 2 * parameters.hopping_ins_l * cos(kx);
        }

        voltage_i = parameters.voltage_r[voltage_step];

        for (int i = 0; i < parameters.num_cor; i++){
            int j = i + parameters.num_cor + parameters.num_ins_left;
            potential_file << j + 1 << "  " << voltage_i <<  "\n";  
            hamiltonian(j, j) = parameters.onsite_cor + voltage_i + 2 * parameters.hopping_cor * cos(kx);
            hamiltonian(j + parameters.chain_length, j + parameters.chain_length) = parameters.onsite_cor + voltage_i + 2 * parameters.hopping_cor * cos(kx);
        }
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
/*
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
        myfile << parameters.energy.at(r) << " " << analytic_gf.at(r).real() << " " << analytic_gf.at(r).imag() << "\n";
        // std::cout << leads.self_energy_left.at(r) << "\n";
    }

    myfile.close();
}
*/
void get_local_gf(Parameters &parameters, std::vector<std::vector<dcomp>> &self_energy_mb, 
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local, int voltage_step, const std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian){

    int n_x = parameters.num_kx_points;
    int n_y = parameters.num_ky_points;

    double num_k_points = n_x * n_y;

    for(int kx_i = 0; kx_i < n_x; kx_i++) {
        for(int ky_i = 0; ky_i < n_y; ky_i++) {

            Interacting_GF gf_interacting(parameters, self_energy_mb,
                leads.at(kx_i).at(ky_i).self_energy_left,
                leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, hamiltonian.at(kx_i).at(ky_i));

            for(int r = 0; r < parameters.steps_myid; r++){
                gf_local.at(r) = (Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
                for(int i = 0; i < 2 * parameters.chain_length; i++){
                    for(int j = 0; j < 2 * parameters.chain_length; j++){
                        gf_local.at(r)(i, j) += gf_interacting.interacting_gf.at(r)(i, j) / num_k_points;
                    }
                }
            }
        }
    }
}

void get_advance_gf(const Parameters &parameters, const Eigen::MatrixXcd &gf_retarded, Eigen::MatrixXcd &gf_advanced){
    for(int i = 0; i < 2 * parameters.chain_length; i++){
        for(int j = 0; j < 2 * parameters.chain_length; j++){
            gf_advanced(i, j) = std::conj(gf_retarded(j, i));
        }
    } 
}

void get_embedding_lesser(const Parameters &parameters, const Eigen::MatrixXcd &self_energy_left, 
    const Eigen::MatrixXcd &self_energy_right, Eigen::MatrixXcd &embedding_self_energy_lesser, int r, int voltage_step){

        embedding_self_energy_lesser(0, 0) = - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) * 
        (self_energy_left(0, 0) - conj(self_energy_left(0, 0)));
        
        embedding_self_energy_lesser(parameters.chain_length, parameters.chain_length) = - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) * 
            (self_energy_left(1, 1) - conj(self_energy_left(1, 1)));

        embedding_self_energy_lesser(parameters.chain_length, 0) = - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) * 
            (self_energy_left(1, 0) - conj(self_energy_left(0, 1)));

        embedding_self_energy_lesser(0, parameters.chain_length) = - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) * 
            (self_energy_left(0, 1) - conj(self_energy_left(1, 0)));


        embedding_self_energy_lesser(parameters.chain_length - 1, parameters.chain_length - 1) += - fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters) * 
            (self_energy_right(0, 0) - conj(self_energy_right(0, 0)));
        
        embedding_self_energy_lesser(2 * parameters.chain_length - 1, 2 * parameters.chain_length - 1) += - fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters) * 
            (self_energy_right(1, 1) - conj(self_energy_right(1, 1)));

        embedding_self_energy_lesser(2 * parameters.chain_length  - 1, parameters.chain_length - 1) += - fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters) * 
            (self_energy_right(1, 0) - conj(self_energy_right(0, 1)));

        embedding_self_energy_lesser(parameters.chain_length  - 1, 2 * parameters.chain_length - 1) += - fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters) * 
            (self_energy_right(0, 1) - conj(self_energy_right(1, 0)));

}

void get_gf_lesser_non_eq(const Parameters &parameters, const std::vector<Eigen::MatrixXcd> &gf_retarded, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const std::vector<Eigen::MatrixXcd> &self_energy_left, 
    const std::vector<Eigen::MatrixXcd> &self_energy_right, std::vector<Eigen::MatrixXcd> &gf_lesser, int voltage_step){

    //std::ofstream embedding_file;
    //std::ostringstream oss;
	//oss << "textfiles/" << parameters.myid << ".embedding.txt";
	//std::string var = oss.str();
	//embedding_file.open(var);

    for(int r = 0; r < parameters.steps_myid; r++) {
        gf_lesser.at(r) = (Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
    }
    //I commented this out cause the wide band limit should stop any bound states.
    //Eigen::MatrixXcd delta_term = Eigen::MatrixXd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length);
        //std::cout << "The lesser green function is" << "\n";

    Eigen::MatrixXcd delta_term = (Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
			
    for(int r = 0; r < parameters.steps_myid; r++) {   
        delta_term = 2.0 * parameters.j1 * parameters.delta_gf * fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)), parameters) 
            * gf_retarded.at(r) * gf_retarded.at(r).adjoint();


        Eigen::MatrixXcd embedding_self_energy = Eigen::MatrixXd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length);
        get_embedding_lesser(parameters, self_energy_left.at(r), self_energy_right.at(r), embedding_self_energy, r + parameters.start.at(parameters.myid), voltage_step);
        
	    //embedding_file << parameters.energy.at(r + parameters.start.at(parameters.myid)) << "  " << embedding_self_energy(0, 0).real() << "   " << embedding_self_energy(0, 0).imag() << "   "
		//    << embedding_self_energy(1, 1).real() << "   " << embedding_self_energy(1, 1).imag() << " \n";    

        Eigen::MatrixXcd gf_advanced = Eigen::MatrixXd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length);

        get_advance_gf(parameters, gf_retarded.at(r), gf_advanced); 
 
        for(int i = 0; i < 2 * parameters.chain_length; i++) {
            for(int j = 0; j < 2 * parameters.chain_length; j++) {  
                for(int k = 0; k < 2 * parameters.chain_length; k++){
                    for(int m = 0; m < 2 * parameters.chain_length; m++){
                        if (m == k){
                            gf_lesser.at(r)(i, j) +=  gf_retarded.at(r)(i, k) * (self_energy_mb_lesser.at(k).at(r) + embedding_self_energy(k, m))
                                * gf_advanced(m, j);
                        } else {
                            gf_lesser.at(r)(i, j) +=  gf_retarded.at(r)(i, k) * (embedding_self_energy(k, m)) 
                                * gf_advanced(m, j);
                        }
                    }
                } 
            }
        }
        gf_lesser.at(r) = gf_lesser.at(r) + delta_term;
    }
    //embedding_file.close();
}

double get_gf_lesser_fd(const Parameters &parameters, const std::vector<Eigen::MatrixXcd> &gf_retarded, const std::vector<Eigen::MatrixXcd> &gf_lesser){
    std::vector<Eigen::MatrixXcd> gf_fd(parameters.steps, Eigen::MatrixXd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length)); 
    double difference = - std::numeric_limits<double>::infinity();
	double old_difference = 0;

	for (int r = 0; r < parameters.steps_myid; r++) {
		for (int i = 0; i < 2 * parameters.chain_length; i++) {
			for (int j = 0; j < 2 * parameters.chain_length; j++) {

                gf_fd.at(r)(i, j) = - 1.0 * fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)), parameters) *
                    (gf_retarded.at(r)(i, j) - std::conj(gf_retarded.at(r)(j, i)));

				old_difference = abs(gf_fd.at(r)(i, j).imag() - gf_lesser.at(r)(i, j).imag());
				//std::cout << real_difference << "  " << imag_difference << "  "  << difference << "\n";
				difference = std::max(difference, old_difference);
				old_difference = difference;
                
			}
		}
    }
    return difference;
}

void get_local_gf_r_and_lesser(const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, std::vector<Eigen::MatrixXcd> &gf_local, 
    std::vector<Eigen::MatrixXcd> &gf_local_lesser, const int voltage_step, const std::vector<std::vector<Eigen::MatrixXcd>> &hamiltonian){

    for(int r = 0; r < parameters.steps_myid; r++){
        gf_local.at(r) = (Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
        gf_local_lesser.at(r) = (Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length));
    }

    int n_x = parameters.num_kx_points, n_y = parameters.num_ky_points;

    std::vector<Eigen::MatrixXcd> gf_lesser(parameters.steps_myid, Eigen::MatrixXcd::Zero(2 * parameters.chain_length, 2 * parameters.chain_length)); 
    double num_k_points = n_x * n_y;
    for(int kx_i = 0; kx_i < n_x; kx_i++) {
        for(int ky_i = 0; ky_i < n_y; ky_i++) {
            Interacting_GF gf_interacting(parameters, self_energy_mb,
                leads.at(kx_i).at(ky_i).self_energy_left,
                leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, hamiltonian.at(kx_i).at(ky_i));    

            get_gf_lesser_non_eq(parameters, gf_interacting.interacting_gf, 
                self_energy_mb_lesser, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
                gf_lesser, voltage_step);

            for(int r = 0; r < parameters.steps_myid; r++){
                for(int i = 0; i < 2 * parameters.chain_length; i++){
                    for(int j = 0; j < 2 * parameters.chain_length; j++){
                        gf_local.at(r)(i, j) += gf_interacting.interacting_gf.at(r)(i, j) / num_k_points;
                        gf_local_lesser.at(r)(i, j) += gf_lesser.at(r)(i, j) / num_k_points;
                    }
                }
            }     
        }
    }
}

