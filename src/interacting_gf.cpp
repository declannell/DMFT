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
    this->interacting_gf.resize(parameters.steps_myid, Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
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
        inverse_gf = Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);

        for (int i = 0; i < 4 *parameters.chain_length; i++){
            for (int j = 0; j < 4 * parameters.chain_length; j++){
                if(i == j){
                    inverse_gf(i, j) = parameters.energy.at(r + parameters.start.at(parameters.myid)) + parameters.j1 * parameters.delta_gf - hamiltonian(i, j) - self_energy_mb.at(i).at(r);
                } else {
                    inverse_gf(i, j) = - hamiltonian(i, j);
                }
            }
        }

        inverse_gf(0, 0) -= self_energy_left.at(r)(0, 0);
        inverse_gf(parameters.chain_length, parameters.chain_length) -= self_energy_left.at(r)(1, 1);
        inverse_gf(2 * parameters.chain_length, 2 * parameters.chain_length) -= self_energy_left.at(r)(2, 2);
        inverse_gf(3 * parameters.chain_length, 3 * parameters.chain_length) -= self_energy_left.at(r)(3, 3);

        inverse_gf(parameters.chain_length - 1, parameters.chain_length - 1) -= self_energy_right.at(r)(0, 0);
        inverse_gf(2 * parameters.chain_length - 1, 2 * parameters.chain_length - 1) -= self_energy_right.at(r)(1, 1);
        inverse_gf(3 * parameters.chain_length - 1, 3 * parameters.chain_length - 1) -= self_energy_right.at(r)(2, 2);
        inverse_gf(4 * parameters.chain_length - 1, 4 * parameters.chain_length - 1) -= self_energy_right.at(r)(3, 3);

        //std::cout << inverse_gf << "\n" << std::endl;

        if(parameters.wbl_approx != true){
            std::cout << "this needs to be fixed if the wba isn't true \n";
            exit(1);
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

    for (int i = 0; i < parameters.chain_length - 1; i++) {
        hamiltonian(i, i + 1) = parameters.hopping_cor;
        hamiltonian(i + 1, i) = parameters.hopping_cor;

        hamiltonian(i + parameters.chain_length, i + 1 + parameters.chain_length) = parameters.hopping_cor;
        hamiltonian(i + 1 + parameters.chain_length, i + parameters.chain_length) = parameters.hopping_cor;

        hamiltonian(i + 2 * parameters.chain_length, i + 1 + 2 * parameters.chain_length) = parameters.hopping_cor;
        hamiltonian(i + 1 + 2 * parameters.chain_length, i + 2 * parameters.chain_length) = parameters.hopping_cor;

        hamiltonian(i + 3 * parameters.chain_length, i + 1 + 3 * parameters.chain_length) = parameters.hopping_cor;
        hamiltonian(i + 1 + 3 * parameters.chain_length, i + 3 * parameters.chain_length) = parameters.hopping_cor;
    }

    for (int i = 0; i < parameters.chain_length; i++){
        if (parameters.num_kx_points != 1) {
            hamiltonian(i, i + parameters.chain_length) = parameters.hopping_cor * (1.0 + exp(parameters.j1 * kx)); //this is the block (1, 2)   
            hamiltonian(i + parameters.chain_length, i) = parameters.hopping_cor * (1.0 + exp(- parameters.j1 * kx)); //this is the block (2, 1)     
            hamiltonian(i + 3 * parameters.chain_length, i + 2 * parameters.chain_length) = parameters.hopping_cor * (1.0 + exp(- parameters.j1 * kx)); //this is the block (4, 3)   
            hamiltonian(i + 2 * parameters.chain_length, i + 3 * parameters.chain_length) = parameters.hopping_cor * (1.0 + exp(parameters.j1 * kx)); //this is the block (3, 4)           
        } else {
            hamiltonian(i, i + parameters.chain_length) = parameters.hopping_cor; //this is the block (1, 2)   
            hamiltonian(i + parameters.chain_length, i) = parameters.hopping_cor; //this is the block (2, 1)     
            hamiltonian(i + 3 * parameters.chain_length, i + 2 * parameters.chain_length) = parameters.hopping_cor; //this is the block (4, 3)   
            hamiltonian(i + 2 * parameters.chain_length, i + 3 * parameters.chain_length) = parameters.hopping_cor;
        }

        if (parameters.num_ky_points != 1) {
            hamiltonian(i, i + 2 * parameters.chain_length) = parameters.hopping_cor * (1.0 + exp(- parameters.j1 * ky)); //this is the block (1, 3)
            hamiltonian(i + 2 * parameters.chain_length, i) = parameters.hopping_cor * (1.0 + exp(parameters.j1 * ky)); //this is the block (3, 1)        
            hamiltonian(i + 3 * parameters.chain_length, i + parameters.chain_length) = parameters.hopping_cor * (1.0 + exp(parameters.j1 * ky)); //this is the block (4, 2)   
            hamiltonian(i + parameters.chain_length, i + 3 * parameters.chain_length) = parameters.hopping_cor * (1.0 + exp(- parameters.j1 * ky)); //this is the block (2, 4) 
        } else {
            hamiltonian(i, i + 2 * parameters.chain_length) = parameters.hopping_cor; //this is the block (1, 3)
            hamiltonian(i + 2 * parameters.chain_length, i) = parameters.hopping_cor; //this is the block (3, 1)        
            hamiltonian(i + 3 * parameters.chain_length, i + parameters.chain_length) = parameters.hopping_cor; //this is the block (4, 2)   
            hamiltonian(i + parameters.chain_length, i + 3 * parameters.chain_length) = parameters.hopping_cor; //this is the block (2, 4)
        }
    }

    //if (parameters.myid == 0) {
    //    std::cout << "initialised the hoppings for second layer \n";
    //}
    if (parameters.ins_metal_ins == true){
        int num_ins = parameters.num_ins_left + parameters.num_ins_right;      
        double delta_v = potential_bias / (double)(num_ins + 1.0);
        if (parameters.num_ins_left != parameters.num_ins_right) {
            std::cout << "you need to change how the voltage drop occurs in the hamiltonian function \n";
            exit(1);
        }

        //std::cout << "failed here 5 \n"; 
        for (int i = 0; i < parameters.num_ins_left; i++){
            voltage_i = parameters.voltage_l[voltage_step] - (double)(i + 1) * delta_v;
            potential_file << i + 1 << "  " << voltage_i <<  "\n"; 
            //this is the top left
            hamiltonian(i, i) = parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l + voltage_i;
            //this is the top right
            hamiltonian(i + parameters.chain_length, i + parameters.chain_length) = 
              -1 * (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l) + voltage_i;
            //this is the bottom left
            hamiltonian(i + 2 * parameters.chain_length, i + 2 * parameters.chain_length) = 
              -1 * (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l) + voltage_i;
            //this is the bottom right
            hamiltonian(i + 3 * parameters.chain_length, i + 3 * parameters.chain_length) = 
               (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l) + voltage_i;             
        }

        if(parameters.num_ins_left == 0 && parameters.num_ins_right != 0){
            voltage_i = parameters.voltage_l[voltage_step];
        } else if (parameters.num_ins_right == 0 && parameters.num_ins_left != 0){
            voltage_i = parameters.voltage_r[voltage_step];
        } else if (parameters.num_ins_right == 0 && parameters.num_ins_left == 0){
            voltage_i = (parameters.voltage_l[voltage_step] + parameters.voltage_r[voltage_step]) * 0.5;
        } else { 
            voltage_i -= - delta_v * 0.5;
            //the voltage is constant in the correlated metal
        }

        //std::cout << "The voltage on the correlated atom is " << voltage_i << std::endl;

        for (int i = 0; i < parameters.num_cor; i++){
            int j = i + parameters.num_ins_left;
            potential_file << j + 1 << "  " << voltage_i <<  "\n";          
            hamiltonian(j, j) = parameters.onsite_cor + voltage_i;
            hamiltonian(j + parameters.chain_length, j + parameters.chain_length) = parameters.onsite_cor + voltage_i;
            hamiltonian(j + 2 * parameters.chain_length, j + 2 * parameters.chain_length) = parameters.onsite_cor + voltage_i;
            hamiltonian(j + 3 * parameters.chain_length, j + 3 * parameters.chain_length) = parameters.onsite_cor + voltage_i;
        }

        for (int i = 0; i < parameters.num_ins_right; i++){
            int j = i + parameters.num_cor + parameters.num_ins_left;
            voltage_i = parameters.voltage_l[voltage_step] - (double)(i + parameters.num_ins_left + 1) * delta_v;
            potential_file << j + 1 << "  " << voltage_i <<  "\n";  
            hamiltonian(j, j) = -(parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r) + voltage_i;
            hamiltonian(j + parameters.chain_length, j + parameters.chain_length) =  (parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r)
                + voltage_i;
            hamiltonian(j + 2 * parameters.chain_length, j + 2 * parameters.chain_length) =  (parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r)
                + voltage_i;

            hamiltonian(j + 3 * parameters.chain_length, j + 3 * parameters.chain_length) =  - (parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r)
                + voltage_i;
        }
    } else { //this is the metal/ins/metal structure

        int num_ins = parameters.num_ins_left;
        
        double delta_v = potential_bias / (double)(num_ins + 1.0);
        //std::cout << delta_v << std::endl;
        //std::cout << "failed here 5 \n"; 
        voltage_i = parameters.voltage_l[voltage_step];
        for (int i = 0; i < parameters.num_cor; i++){
            potential_file << i + 1 << "  " << voltage_i <<  "\n";  
            hamiltonian(i, i) = parameters.onsite_cor + voltage_i;
            hamiltonian(i + parameters.chain_length, i + parameters.chain_length) = parameters.onsite_cor + voltage_i;
            hamiltonian(i + 2 * parameters.chain_length, i + 2 * parameters.chain_length) = parameters.onsite_cor + voltage_i;
            hamiltonian(i + 3 * parameters.chain_length, i + 3 * parameters.chain_length) = parameters.onsite_cor + voltage_i;          
        }

        //std::cout << "The voltage on the correlated atom is " << voltage_i << std::endl;

        for (int i = 0; i < parameters.num_ins_left; i++){
            int j = i + parameters.num_cor;
            voltage_i = parameters.voltage_l[voltage_step] - (double)(i + 1) * delta_v;
            //std::cout << voltage_i << std::endl;
            potential_file << j + 1 << "  " << voltage_i <<  "\n";          
            hamiltonian(j, j) = parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l + voltage_i;
            hamiltonian(j + parameters.chain_length, j + parameters.chain_length) = - (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l) + voltage_i;
            hamiltonian(j + 2 * parameters.chain_length, j + 2 * parameters.chain_length) =  - (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l)
                + voltage_i;
            hamiltonian(j + 3 * parameters.chain_length, j + 3 * parameters.chain_length) =  - (parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r)
                + voltage_i;  
        }

        voltage_i = parameters.voltage_r[voltage_step];

        for (int i = 0; i < parameters.num_cor; i++){
            int j = i + parameters.num_cor + parameters.num_ins_left;
            potential_file << j + 1 << "  " << voltage_i <<  "\n";  
            hamiltonian(j, j) = parameters.onsite_cor + voltage_i;
            hamiltonian(j + parameters.chain_length, j + parameters.chain_length) = parameters.onsite_cor + voltage_i;
            hamiltonian(j + 2 * parameters.chain_length, j + 2 * parameters.chain_length) = parameters.onsite_cor + voltage_i;
            hamiltonian(j + 3 * parameters.chain_length, j + 3 * parameters.chain_length) = parameters.onsite_cor + voltage_i;
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
                gf_local.at(r) = (Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
                for(int i = 0; i < 4 * parameters.chain_length; i++){
                    for(int j = 0; j < 4 * parameters.chain_length; j++){
                        gf_local.at(r)(i, j) += gf_interacting.interacting_gf.at(r)(i, j) / num_k_points;
                    }
                }
            }
        }
    }
}

void get_advance_gf(const Parameters &parameters, const Eigen::MatrixXcd &gf_retarded, Eigen::MatrixXcd &gf_advanced){
    for(int i = 0; i < 4 * parameters.chain_length; i++){
        for(int j = 0; j < 4 * parameters.chain_length; j++){
            gf_advanced(i, j) = std::conj(gf_retarded(j, i));
        }
    } 
}

void get_embedding_lesser(const Parameters &parameters, const Eigen::MatrixXcd &self_energy_left, 
    const Eigen::MatrixXcd &self_energy_right, Eigen::MatrixXcd &embedding_self_energy_lesser, int r, int voltage_step){

        if (parameters.wbl_approx != true) {
            std::cout << "you need to rewrite the embedding lesser self energy \n";
            exit(1);
        }
        embedding_self_energy_lesser(0, 0) = - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) * 
        (self_energy_left(0, 0) - conj(self_energy_left(0, 0)));
        
        embedding_self_energy_lesser(parameters.chain_length, parameters.chain_length) = - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) * 
            (self_energy_left(1, 1) - conj(self_energy_left(1, 1)));

        embedding_self_energy_lesser(2 * parameters.chain_length, 2 * parameters.chain_length) = - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) * 
            (self_energy_left(2, 2) - conj(self_energy_left(2, 2)));

        embedding_self_energy_lesser(3 * parameters.chain_length, 3 * parameters.chain_length) = - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) * 
            (self_energy_left(3, 3) - conj(self_energy_left(3, 3)));


        embedding_self_energy_lesser(parameters.chain_length - 1, parameters.chain_length - 1) += - fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters) * 
            (self_energy_right(0, 0) - conj(self_energy_right(0, 0)));
        
        embedding_self_energy_lesser(2 * parameters.chain_length - 1, 2 * parameters.chain_length - 1) += - fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters) * 
            (self_energy_right(1, 1) - conj(self_energy_right(1, 1)));

        embedding_self_energy_lesser(3 * parameters.chain_length  - 1, 3 * parameters.chain_length - 1) += - fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters) * 
            (self_energy_right(2, 2) - conj(self_energy_right(2, 2)));

        embedding_self_energy_lesser(4 * parameters.chain_length  - 1, 4 * parameters.chain_length - 1) += - fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters) * 
            (self_energy_right(3, 3) - conj(self_energy_right(3, 3)));
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
        gf_lesser.at(r) = (Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
    }
    //I commented this out cause the wide band limit should stop any bound states.
    Eigen::MatrixXcd delta_term = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
        //std::cout << "The lesser green function is" << "\n";

			
    for(int r = 0; r < parameters.steps_myid; r++) {   
        delta_term = 2.0 * parameters.j1 * parameters.delta_gf * fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)), parameters) 
            * gf_retarded.at(r) * gf_retarded.at(r).adjoint();




        Eigen::MatrixXcd embedding_self_energy = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
        get_embedding_lesser(parameters, self_energy_left.at(r), self_energy_right.at(r), embedding_self_energy, r + parameters.start.at(parameters.myid), voltage_step);
        


	    //embedding_file << parameters.energy.at(r + parameters.start.at(parameters.myid)) << "  " << embedding_self_energy(0, 0).real() << "   " << embedding_self_energy(0, 0).imag() << "   "
		//    << embedding_self_energy(1, 1).real() << "   " << embedding_self_energy(1, 1).imag() << " \n";    

        Eigen::MatrixXcd gf_advanced = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);

        get_advance_gf(parameters, gf_retarded.at(r), gf_advanced); 
 
        for(int i = 0; i < 4 * parameters.chain_length; i++) {
            for(int j = 0; j < 4 * parameters.chain_length; j++) {  
                for(int k = 0; k < 4 * parameters.chain_length; k++){
                    for(int m = 0; m < 4 * parameters.chain_length; m++){
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
		for (int i = 0; i < 4 * parameters.chain_length; i++) {
			for (int j = 0; j < 4 * parameters.chain_length; j++) {

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
        gf_local.at(r) = (Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
        gf_local_lesser.at(r) = (Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
    }

    int n_x = parameters.num_kx_points, n_y = parameters.num_ky_points;

    std::vector<Eigen::MatrixXcd> gf_lesser(parameters.steps_myid, Eigen::MatrixXcd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length)); 
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
                for(int i = 0; i < 4 * parameters.chain_length; i++){
                    for(int j = 0; j < 4 * parameters.chain_length; j++){
                        gf_local.at(r)(i, j) += gf_interacting.interacting_gf.at(r)(i, j) / num_k_points;
                        gf_local_lesser.at(r)(i, j) += gf_lesser.at(r)(i, j) / num_k_points;
                    }
                }
            }     
        }
    }
}

