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

Interacting_GF::Interacting_GF(const Parameters &parameters, const std::vector<std::vector<dcomp>> &self_energy_mb, const MatrixVectorType 
    &self_energy_left, const MatrixVectorType &self_energy_right, const int voltage_step, const  MatrixType &hamiltonian)
{
    this->interacting_gf.resize(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
    get_interacting_gf(parameters, hamiltonian, self_energy_mb, self_energy_left, self_energy_right, voltage_step);
}

void Interacting_GF::get_interacting_gf(const Parameters &parameters, const MatrixType& hamiltonian, const std::vector<std::vector<dcomp>> &self_energy_mb, 
        const MatrixVectorType &self_energy_left, const MatrixVectorType &self_energy_right, const int voltage_step){
    MatrixType inverse_gf;

    for(int r = 0; r < parameters.steps_myid; r++){
        int y = r + parameters.start.at(parameters.myid);
        inverse_gf = MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
        Eigen::MatrixXcd energy = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);

        for (int i = 0; i < 4 *parameters.chain_length; i++) energy(i, i) = parameters.energy.at(y) + parameters.j1 * parameters.delta_gf - self_energy_mb.at(i).at(r);
            
        inverse_gf = energy - hamiltonian - self_energy_left.at(r) - self_energy_right.at(r);
        this->interacting_gf.at(r) = inverse_gf.inverse();
    }
    //dos_file_non_int.close();
}


void get_hamiltonian(Parameters const &parameters, const int voltage_step, const double kx, const double ky, MatrixType &hamiltonian, int spin){
    
    std::ofstream potential_file;
	std::ostringstream ossgf;
	ossgf << voltage_step << ".potential.dat";
	std::string var = ossgf.str();
    potential_file.open(var);
    potential_file << -5 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -4 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -3 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -2 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << -1 << "  " << parameters.voltage_l[voltage_step] <<  "\n";
    potential_file << 0 << "  " << parameters.voltage_l[voltage_step] <<  "\n";

    double voltage_i;

    double magnetic_field;

    //this will be zero if I have no external_magentic field
    if (spin == 1) {//creates a positive term for adding onto the the onsite energy for spin up
        magnetic_field = + parameters.magnetic_field / 2;
    } else {//creates a negative term for adding onto the the onsite energy for spin down
        magnetic_field = - parameters.magnetic_field / 2;
    }

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
        double delta_v =  parameters.voltage_l[voltage_step] / (double)(parameters.num_ins_left + 1.0);
        //std::cout << delta_v << std::endl;
        if (parameters.num_ins_left != parameters.num_ins_right) {
            std::cout << "you need to change how the voltage drop occurs in the hamiltonian function \n";
            exit(1);
        }

        if (parameters.half_metal == 1 && spin == 1) {
            for (int i = 0; i < parameters.num_ins_left; i++){
                voltage_i = parameters.voltage_l[voltage_step] - (double)(i + 1) * delta_v;
                potential_file << i + 1 << "  " << voltage_i <<  "\n"; 
                //this is the top left
                hamiltonian(i, i) = parameters.onsite_cor + voltage_i + magnetic_field;
                //this is the top right
                hamiltonian(i + parameters.chain_length, i + parameters.chain_length) = 
                  parameters.onsite_cor + voltage_i + magnetic_field;
                //this is the bottom left
                hamiltonian(i + 2 * parameters.chain_length, i + 2 * parameters.chain_length) = 
                  parameters.onsite_cor + voltage_i + magnetic_field;
                //this is the bottom right
                hamiltonian(i + 3 * parameters.chain_length, i + 3 * parameters.chain_length) = 
                   parameters.onsite_cor + voltage_i + magnetic_field;             
            }
        } else {
            for (int i = 0; i < parameters.num_ins_left; i++){
                voltage_i = parameters.voltage_l[voltage_step] - (double)(i + 1) * delta_v;
                potential_file << i + 1 << "  " << voltage_i <<  "\n"; 
                //this is the top left
                hamiltonian(i, i) = parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l + voltage_i + magnetic_field;
                //this is the top right
                hamiltonian(i + parameters.chain_length, i + parameters.chain_length) = 
                  -1 * (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l) + voltage_i + magnetic_field;
                //this is the bottom left
                hamiltonian(i + 2 * parameters.chain_length, i + 2 * parameters.chain_length) = 
                  -1 * (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l) + voltage_i + magnetic_field;
                //this is the bottom right
                hamiltonian(i + 3 * parameters.chain_length, i + 3 * parameters.chain_length) = 
                   (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l) + voltage_i + magnetic_field;             
            }
        }
        //std::cout << "failed here 5 \n";

        voltage_i = 0;

        //std::cout << "The voltage on the correlated atom is " << voltage_i << std::endl;

        for (int i = 0; i < parameters.num_cor; i++){
            int j = i + parameters.num_ins_left;
            potential_file << j + 1 << "  " << voltage_i <<  "\n";          
            hamiltonian(j, j) = parameters.onsite_cor + voltage_i + magnetic_field;
            hamiltonian(j + parameters.chain_length, j + parameters.chain_length) = parameters.onsite_cor + voltage_i + magnetic_field;
            hamiltonian(j + 2 * parameters.chain_length, j + 2 * parameters.chain_length) = parameters.onsite_cor + voltage_i + magnetic_field;
            hamiltonian(j + 3 * parameters.chain_length, j + 3 * parameters.chain_length) = parameters.onsite_cor + voltage_i + magnetic_field;
        }

        if (parameters.half_metal == 1 && spin == 1) {
            for (int i = 0; i < parameters.num_ins_right; i++){
                int j = i + parameters.num_cor + parameters.num_ins_left;
                voltage_i = - (double)(i + 1) * delta_v;
                potential_file << j + 1 << "  " << voltage_i <<  "\n";  
                hamiltonian(j, j) = parameters.onsite_cor + voltage_i + magnetic_field;
                hamiltonian(j + parameters.chain_length, j + parameters.chain_length) =  parameters.onsite_cor
                    + voltage_i + magnetic_field;
                hamiltonian(j + 2 * parameters.chain_length, j + 2 * parameters.chain_length) =  parameters.onsite_cor
                    + voltage_i + magnetic_field;

                hamiltonian(j + 3 * parameters.chain_length, j + 3 * parameters.chain_length) =  parameters.onsite_cor
                    + voltage_i + magnetic_field;
            }
        } else {
            for (int i = 0; i < parameters.num_ins_right; i++){
                int j = i + parameters.num_cor + parameters.num_ins_left;
                voltage_i = - (double)(i + 1) * delta_v;
                potential_file << j + 1 << "  " << voltage_i <<  "\n";  
                hamiltonian(j, j) = -(parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r) + voltage_i + magnetic_field;
                hamiltonian(j + parameters.chain_length, j + parameters.chain_length) =  (parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r)
                    + voltage_i + magnetic_field;
                hamiltonian(j + 2 * parameters.chain_length, j + 2 * parameters.chain_length) =  (parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r)
                    + voltage_i + magnetic_field;

                hamiltonian(j + 3 * parameters.chain_length, j + 3 * parameters.chain_length) =  - (parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r)
                    + voltage_i + magnetic_field;
            }
        }
    } else { //this is the metal/ins/metal structure

        int num_ins = parameters.num_ins_left;
        
        double delta_v = parameters.voltage_l[voltage_step] * 2 / (double)(num_ins + 1.0);
        //std::cout << delta_v << std::endl;
        //std::cout << "failed here 5 \n"; 
        voltage_i = parameters.voltage_l[voltage_step];
        for (int i = 0; i < parameters.num_cor; i++){
            potential_file << i + 1 << "  " << voltage_i <<  "\n";  
            hamiltonian(i, i) = parameters.onsite_cor + voltage_i + magnetic_field;
            hamiltonian(i + parameters.chain_length, i + parameters.chain_length) = parameters.onsite_cor + voltage_i + magnetic_field;
            hamiltonian(i + 2 * parameters.chain_length, i + 2 * parameters.chain_length) = parameters.onsite_cor + voltage_i + magnetic_field;
            hamiltonian(i + 3 * parameters.chain_length, i + 3 * parameters.chain_length) = parameters.onsite_cor + voltage_i + magnetic_field;          
        }

        //std::cout << "The voltage on the correlated atom is " << voltage_i << std::endl;
        if (parameters.half_metal == 1 && spin == 1) {
            for (int i = 0; i < parameters.num_ins_left; i++){
                int j = i + parameters.num_cor;
                voltage_i = parameters.voltage_l[voltage_step] - (double)(i + 1) * delta_v;
                //std::cout << voltage_i << std::endl;
                potential_file << j + 1 << "  " << voltage_i <<  "\n";          
                hamiltonian(j, j) = parameters.onsite_cor + voltage_i + magnetic_field;
                hamiltonian(j + parameters.chain_length, j + parameters.chain_length) = parameters.onsite_cor + voltage_i
                    + magnetic_field;
                hamiltonian(j + 2 * parameters.chain_length, j + 2 * parameters.chain_length) = parameters.onsite_cor
                    + voltage_i + magnetic_field;
                hamiltonian(j + 3 * parameters.chain_length, j + 3 * parameters.chain_length) = parameters.onsite_cor
                    + voltage_i + magnetic_field;  
            }
        } else {
            for (int i = 0; i < parameters.num_ins_left; i++){
                int j = i + parameters.num_cor;
                voltage_i = parameters.voltage_l[voltage_step] - (double)(i + 1) * delta_v;
                //std::cout << voltage_i << std::endl;
                potential_file << j + 1 << "  " << voltage_i <<  "\n";          
                hamiltonian(j, j) = parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l + voltage_i + magnetic_field;
                hamiltonian(j + parameters.chain_length, j + parameters.chain_length) = - (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l) + voltage_i
                    + magnetic_field;
                hamiltonian(j + 2 * parameters.chain_length, j + 2 * parameters.chain_length) =  - (parameters.onsite_ins_l - 2 * (i % 2) * parameters.onsite_ins_l)
                    + voltage_i + magnetic_field;
                hamiltonian(j + 3 * parameters.chain_length, j + 3 * parameters.chain_length) =   (parameters.onsite_ins_r - 2 * (i % 2) * parameters.onsite_ins_r)
                    + voltage_i + magnetic_field;  
            }
        }

        voltage_i = parameters.voltage_r[voltage_step];

        for (int i = 0; i < parameters.num_cor; i++){
            int j = i + parameters.num_cor + parameters.num_ins_left;
            potential_file << j + 1 << "  " << voltage_i <<  "\n";  
            hamiltonian(j, j) = parameters.onsite_cor + voltage_i + magnetic_field;
            hamiltonian(j + parameters.chain_length, j + parameters.chain_length) = parameters.onsite_cor + voltage_i + magnetic_field;
            hamiltonian(j + 2 * parameters.chain_length, j + 2 * parameters.chain_length) = parameters.onsite_cor + voltage_i + magnetic_field;
            hamiltonian(j + 3 * parameters.chain_length, j + 3 * parameters.chain_length) = parameters.onsite_cor + voltage_i + magnetic_field;
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
    std::vector<std::vector<EmbeddingSelfEnergy>> &leads, MatrixVectorType &gf_local, int voltage_step, const std::vector<MatrixVectorType> &hamiltonian){

    int n_x = parameters.num_kx_points;
    int n_y = parameters.num_ky_points;

    double num_k_points = n_x * n_y;

    for(int kx_i = 0; kx_i < n_x; kx_i++) {
        for(int ky_i = 0; ky_i < n_y; ky_i++) {

            Interacting_GF gf_interacting(parameters, self_energy_mb,
                leads.at(kx_i).at(ky_i).self_energy_left,
                leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, hamiltonian.at(kx_i).at(ky_i));

            for(int r = 0; r < parameters.steps_myid; r++){
                gf_local.at(r) = (MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
                for(int i = 0; i < 4 * parameters.chain_length; i++){
                    for(int j = 0; j < 4 * parameters.chain_length; j++){
                        gf_local.at(r)(i, j) += gf_interacting.interacting_gf.at(r)(i, j) / num_k_points;
                    }
                }
            }
        }
    }
}

void get_advance_gf(const Parameters &parameters, const MatrixType &gf_retarded, MatrixType &gf_advanced){
    for(int i = 0; i < 4 * parameters.chain_length; i++){
        for(int j = 0; j < 4 * parameters.chain_length; j++){
            gf_advanced(i, j) = std::conj(gf_retarded(j, i));
        }
    } 
}

void get_embedding_lesser(const Parameters &parameters, const MatrixType &self_energy_left, 
    const MatrixType &self_energy_right, MatrixType &embedding_self_energy_lesser, int r, int voltage_step){

        embedding_self_energy_lesser = - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters) * 
        (self_energy_left - self_energy_left.adjoint()) - fermi_function(parameters.energy.at(r) - parameters.voltage_r[voltage_step], parameters) * 
        (self_energy_right - self_energy_right.adjoint());
}

void get_embedding_greater(const Parameters &parameters, const MatrixType &self_energy_left, 
    const MatrixType &self_energy_right, MatrixType &embedding_self_energy_greater, int r, int voltage_step){

        embedding_self_energy_greater = (1.0 - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters)) * 
        (self_energy_left - self_energy_left.adjoint()) 
        - (1.0 - fermi_function(parameters.energy.at(r) - parameters.voltage_l[voltage_step], parameters)) * 
        (self_energy_right - self_energy_right.adjoint());
}


void get_gf_lesser_non_eq(const Parameters &parameters, const MatrixVectorType &gf_retarded, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const MatrixVectorType &self_energy_left, 
    const MatrixVectorType &self_energy_right, MatrixVectorType &gf_lesser, int voltage_step){

    for(int r = 0; r < parameters.steps_myid; r++) {
        gf_lesser.at(r) = (MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
    }
			
    //if (parameters.wbl_approx == 1) { //we don't need the diagonal elements of the lesser green function as embedding self energies are diagonal.
    //    for(int r = 0; r < parameters.steps_myid; r++) {   
    //        int y = r + parameters.start.at(parameters.myid);
    //        MatrixType embedding_self_energy = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
    //        get_embedding_lesser(parameters, self_energy_left.at(r), self_energy_right.at(r), embedding_self_energy, y, voltage_step);
    //        //std::cout << embedding_self_energy(0, 0) << " " << self_energy_left.at(r)(0, 0) << std::endl;
    //        for(int i = 0; i < 4 * parameters.chain_length; i++) {
    //            for(int m = 0; m < 4 * parameters.chain_length; m++){//I don't need bound state correction in the wide band limit.
    //                gf_lesser.at(r)(i, i) +=  gf_retarded.at(r)(i, m) * (self_energy_mb_lesser.at(m).at(r) + embedding_self_energy(m, m))
    //                    * std::conj(gf_retarded.at(r)(i, m));
    //            }
    //            //gf_lesser.at(r)(i, i) = - 2.0 * parameters.j1 * fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)), parameters) 
    //            //    * (gf_retarded.at(r)(i, i)).imag();
    //        }
    //    }
    //} else {//else I need the whole lesser green function
        for(int r = 0; r < parameters.steps_myid; r++) {   //I need the bound state correction to get the correct occupation when not using the wide band approx.
            MatrixType embedding_self_energy = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
            MatrixType mb_lesser_self_energy = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);

            get_embedding_lesser(parameters, self_energy_left.at(r), self_energy_right.at(r), embedding_self_energy, r + parameters.start.at(parameters.myid), voltage_step);

            for(int i = 0; i < 4 * parameters.chain_length; i++) mb_lesser_self_energy(i, i) = self_energy_mb_lesser.at(i).at(r);

            int y = r + parameters.start.at(parameters.myid);
            gf_lesser.at(r) =  gf_retarded.at(r) * (mb_lesser_self_energy + embedding_self_energy) * (gf_retarded.at(r)).adjoint()
                + 2.0 * parameters.j1 * parameters.delta_gf * fermi_function(parameters.energy.at(y), parameters) * gf_retarded.at(r) * (gf_retarded.at(r)).adjoint();
        }
    //}
}

void get_gf_lesser_greater_non_eq(const Parameters &parameters, const MatrixVectorType &gf_retarded, 
    const std::vector<std::vector<dcomp>> &self_energy_mb_lesser, const std::vector<std::vector<dcomp>> &self_energy_mb_greater,
    const MatrixVectorType &self_energy_left, const MatrixVectorType &self_energy_right, MatrixVectorType &gf_lesser,
    MatrixVectorType &gf_greater, int voltage_step){

    for(int r = 0; r < parameters.steps_myid; r++) {
        gf_lesser.at(r) = (MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
        gf_greater.at(r) = (MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
    }
	//		
    //if (parameters.wbl_approx == 1 && parameters.bond_current == 0) { //we don't need the diagonal elements of the lesser green function as embedding self energies are diagonal.
    //    for(int r = 0; r < parameters.steps_myid; r++) {   
    //        MatrixType embedding_self_energy_lesser = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
    //        MatrixType embedding_self_energy_greater = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
    //        get_embedding_lesser(parameters, self_energy_left.at(r), self_energy_right.at(r), embedding_self_energy_lesser, r + parameters.start.at(parameters.myid), voltage_step);
    //        get_embedding_greater(parameters, self_energy_left.at(r), self_energy_right.at(r), embedding_self_energy_greater, r + parameters.start.at(parameters.myid), voltage_step);
    //        for(int i = 0; i < 4 * parameters.chain_length; i++) {
    //            for(int m = 0; m < 4 * parameters.chain_length; m++){
    //                gf_lesser.at(r)(i, i) +=  gf_retarded.at(r)(i, m) * (self_energy_mb_lesser.at(m).at(r) + embedding_self_energy_lesser(m, m))
    //                    * std::conj(gf_retarded.at(r)(i, m));
//
    //                gf_greater.at(r)(i, i) +=  gf_retarded.at(r)(i, m) * (self_energy_mb_greater.at(m).at(r) + embedding_self_energy_greater(m, m))
    //                    * std::conj(gf_retarded.at(r)(i, m));
    //            }
    //        }
    //    }
    //} else {//else I need the whole lesser green function
        for(int r = 0; r < parameters.steps_myid; r++) {   
            MatrixType mb_lesser_self_energy = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
            MatrixType mb_greater_self_energy = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);

            MatrixType embedding_self_energy_lesser = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
            MatrixType embedding_self_energy_greater = Eigen::MatrixXd::Zero(4 * parameters.chain_length, 4 * parameters.chain_length);
            get_embedding_lesser(parameters, self_energy_left.at(r), self_energy_right.at(r), embedding_self_energy_lesser, r + parameters.start.at(parameters.myid), voltage_step);
            get_embedding_greater(parameters, self_energy_left.at(r), self_energy_right.at(r), embedding_self_energy_greater, r + parameters.start.at(parameters.myid), voltage_step);

            for(int i = 0; i < 4 * parameters.chain_length; i++) mb_lesser_self_energy(i, i) = self_energy_mb_lesser.at(i).at(r);
            for(int i = 0; i < 4 * parameters.chain_length; i++) mb_greater_self_energy(i, i) = self_energy_mb_greater.at(i).at(r);

            gf_lesser.at(r) =  gf_retarded.at(r) * (mb_lesser_self_energy + embedding_self_energy_lesser) * (gf_retarded.at(r)).adjoint();
            gf_greater.at(r) =  gf_retarded.at(r) * (mb_lesser_self_energy + embedding_self_energy_greater) * (gf_retarded.at(r)).adjoint();
        }
    //}
}


void get_gf_lesser_fd(const Parameters &parameters, const MatrixVectorType &gf_retarded, MatrixVectorType &gf_lesser){
	for (int r = 0; r < parameters.steps_myid; r++) {
        gf_lesser.at(r) = - 1.0 * fermi_function(parameters.energy.at(r + parameters.start.at(parameters.myid)), parameters) *
            (gf_retarded.at(r) - (gf_retarded.at(r)).adjoint());
    }
}


void get_local_gf_r_and_lesser(const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, MatrixVectorType &gf_local, 
    MatrixVectorType &gf_local_lesser, const int voltage_step, const std::vector<MatrixVectorType> &hamiltonian){

    for(int r = 0; r < parameters.steps_myid; r++){
        gf_local.at(r) = (MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
        gf_local_lesser.at(r) = (MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
    }



    int n_x = parameters.num_kx_points, n_y = parameters.num_ky_points;

    MatrixVectorType gf_lesser(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length)); 
    double num_k_points = n_x * n_y;
    if (parameters.wbl_approx == 1) {
        for(int kx_i = 0; kx_i < n_x; kx_i++) {
            for(int ky_i = 0; ky_i < n_y; ky_i++) {
                Interacting_GF gf_interacting(parameters, self_energy_mb,
                    leads.at(kx_i).at(ky_i).self_energy_left,
                    leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, hamiltonian.at(kx_i).at(ky_i));   

                get_gf_lesser_non_eq(parameters, gf_interacting.interacting_gf, 
                    self_energy_mb_lesser, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
                    gf_lesser, voltage_step);

                for(int r = 0; r < parameters.steps_myid; r++){
                    gf_local.at(r) += gf_interacting.interacting_gf.at(r) * (1.0 / num_k_points);
                    for(int i = 0; i < 4 * parameters.chain_length; i++){
                        gf_local_lesser.at(r)(i, i) += gf_lesser.at(r)(i, i) / num_k_points;
                    }
                }     
            }
        }
    } else if (parameters.wbl_approx == 0) {
        for(int kx_i = 0; kx_i < n_x; kx_i++) {
            for(int ky_i = 0; ky_i < n_y; ky_i++) {
                Interacting_GF gf_interacting(parameters, self_energy_mb,
                    leads.at(kx_i).at(ky_i).self_energy_left,
                    leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, hamiltonian.at(kx_i).at(ky_i));   

                get_gf_lesser_non_eq(parameters, gf_interacting.interacting_gf, 
                    self_energy_mb_lesser, leads.at(kx_i).at(ky_i).self_energy_left, leads.at(kx_i).at(ky_i).self_energy_right,
                    gf_lesser, voltage_step);

                for(int r = 0; r < parameters.steps_myid; r++){
                    gf_local.at(r) += gf_interacting.interacting_gf.at(r) * (1.0 / num_k_points);
                    gf_local_lesser.at(r) += gf_lesser.at(r) * (1.0 / num_k_points);
                }     
            }
        }
    }
}


void get_local_gf_r_greater_lesser(const Parameters &parameters, 
    const std::vector<std::vector<dcomp>> &self_energy_mb, const std::vector<std::vector<dcomp>> &self_energy_mb_lesser,
    const std::vector<std::vector<dcomp>> &self_energy_mb_greater,
    const std::vector<std::vector<EmbeddingSelfEnergy>> &leads, MatrixVectorType &gf_local, 
    MatrixVectorType &gf_local_lesser, MatrixVectorType &gf_local_greater, 
    const int voltage_step, const std::vector<MatrixVectorType> &hamiltonian){

    for(int r = 0; r < parameters.steps_myid; r++){
        gf_local.at(r) = (MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
        gf_local_lesser.at(r) = (MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
        gf_local_greater.at(r) = (MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length));
    }

    int n_x = parameters.num_kx_points, n_y = parameters.num_ky_points;

    MatrixVectorType gf_lesser(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length)),
        gf_greater(parameters.steps_myid, MatrixType::Zero(4 * parameters.chain_length, 4 * parameters.chain_length)); 

    double num_k_points = n_x * n_y;
    if (parameters.wbl_approx == 1) {
        for(int kx_i = 0; kx_i < n_x; kx_i++) {
            for(int ky_i = 0; ky_i < n_y; ky_i++) {
                Interacting_GF gf_interacting(parameters, self_energy_mb,
                    leads.at(kx_i).at(ky_i).self_energy_left,
                    leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, hamiltonian.at(kx_i).at(ky_i));   

                get_gf_lesser_greater_non_eq(parameters, gf_interacting.interacting_gf, self_energy_mb_lesser, self_energy_mb_greater, leads.at(kx_i).at(ky_i).self_energy_left, 
                    leads.at(kx_i).at(ky_i).self_energy_right, gf_lesser, gf_greater, voltage_step);

                for(int r = 0; r < parameters.steps_myid; r++){
                    gf_local.at(r) += gf_interacting.interacting_gf.at(r) * (1.0 / num_k_points);
                    for (int i = 0; i < parameters.chain_length * 4; i++) {
                        gf_local_lesser.at(r)(i, i) += gf_lesser.at(r)(i, i) / num_k_points;
                        gf_local_greater.at(r)(i, i) += gf_greater.at(r)(i, i) / num_k_points;
                    }
                }  
            }
        }
    } else if (parameters.wbl_approx == 0) {
        for(int kx_i = 0; kx_i < n_x; kx_i++) {
            for(int ky_i = 0; ky_i < n_y; ky_i++) {
                Interacting_GF gf_interacting(parameters, self_energy_mb,
                    leads.at(kx_i).at(ky_i).self_energy_left,
                    leads.at(kx_i).at(ky_i).self_energy_right, voltage_step, hamiltonian.at(kx_i).at(ky_i));   

                get_gf_lesser_greater_non_eq(parameters, gf_interacting.interacting_gf, self_energy_mb_lesser, self_energy_mb_greater, leads.at(kx_i).at(ky_i).self_energy_left, 
                    leads.at(kx_i).at(ky_i).self_energy_right, gf_lesser, gf_greater, voltage_step);


                for(int r = 0; r < parameters.steps_myid; r++){
                    gf_local.at(r) += gf_interacting.interacting_gf.at(r) * (1.0 / num_k_points);
                    gf_local_lesser.at(r) += gf_lesser.at(r) * (1.0 / num_k_points);
                    gf_local_greater.at(r) += gf_greater.at(r) * (1.0 / num_k_points);
                }  
            }
        }
    }
}

void get_density_matrix(Parameters &parameters, MatrixVectorType &gf_lesser, dcomp &density_matrix, int i, int j){
    dcomp density_element_myid = 0;
    density_matrix = 0;


    for (int r = 0; r < parameters.steps_myid; r++) {
        density_element_myid += gf_lesser.at(r)(i, j);
    }
    density_element_myid = density_element_myid * parameters.delta_energy / (2.0 * M_PI * parameters.j1);
    MPI_Allreduce(&density_element_myid, &density_matrix, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
}

