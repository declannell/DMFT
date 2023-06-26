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
#include "aim.h"
#include "utilis.h"
#include <limits>

AIM::AIM(const Parameters &parameters, const std::vector<dcomp> &local_gf_retarded, const std::vector<dcomp> &local_gf_lesser, 
    const std::vector<dcomp> &local_gf_greater, const std::vector<dcomp> &self_energy_retarded, const std::vector<dcomp> &self_energy_lesser,
    const std::vector<dcomp> &self_energy_greater, const int voltage_step)
{
    this->impurity_gf_mb_retarded.resize(parameters.steps_myid);
    this->dynamical_field_retarded.resize(parameters.steps_myid);
    this->self_energy_mb_retarded.resize(parameters.steps_myid);
    
    this->self_energy_mb_lesser.resize(parameters.steps_myid);
    this->impurity_gf_mb_lesser.resize(parameters.steps_myid);
    this->dynamical_field_lesser.resize(parameters.steps_myid);
    this->hybridisation_lesser.resize(parameters.steps_myid);
    
    this->fermi_function_eff.resize(parameters.steps_myid);

    if (parameters.impurity_solver == 3) {
        this->self_energy_mb_greater.resize(parameters.steps_myid);
        this->impurity_gf_mb_greater.resize(parameters.steps_myid);
        this->dynamical_field_greater.resize(parameters.steps_myid);
        this->hybridisation_greater.resize(parameters.steps_myid);   
       
        get_impurity_gf_mb(parameters, local_gf_retarded, local_gf_lesser, local_gf_greater);
        get_retarded_dynamical_field(parameters, self_energy_retarded);
        get_lesser_greater_hybridisation(parameters, self_energy_lesser, self_energy_greater);
        get_dynamical_field_lesser_greater(parameters, voltage_step);
    } else {
        get_impurity_gf_mb(parameters, local_gf_retarded, local_gf_lesser);
        get_retarded_dynamical_field(parameters, self_energy_retarded);
        get_lesser_hybridisation(parameters, self_energy_lesser);   
        get_dynamical_field_lesser(parameters, voltage_step);
        get_effective_fermi_function(parameters, voltage_step); 
    }


}

void AIM::get_impurity_gf_mb(const Parameters &parameters, const std::vector<dcomp> &local_gf_retarded, const std::vector<dcomp> &local_gf_lesser)
{
    for (int r = 0; r < parameters.steps_myid; r++){
        this->impurity_gf_mb_retarded.at(r) = local_gf_retarded.at(r);
        this->impurity_gf_mb_lesser.at(r) = local_gf_lesser.at(r).imag();
    }
}

void AIM::get_impurity_gf_mb(const Parameters &parameters, const std::vector<dcomp> &local_gf_retarded, const std::vector<dcomp> &local_gf_lesser,
    const std::vector<dcomp> &local_gf_greater)
{
    for (int r = 0; r < parameters.steps_myid; r++){        
        this->impurity_gf_mb_retarded.at(r) = local_gf_retarded.at(r);
        this->impurity_gf_mb_lesser.at(r) = local_gf_lesser.at(r).imag();
        this->impurity_gf_mb_greater.at(r) = local_gf_greater.at(r).imag();
        //write_to_file(parameters, impurity_gf_mb_lesser, "impurity_gf_mb_lesser", 0);
        //write_to_file(parameters, impurity_gf_mb_greater, "impurity_gf_mb_greater", 0);
        //write_to_file(parameters, impurity_gf_mb_retarded, "impurity_gf_mb_retarded", 0);
    }
}

void AIM::get_retarded_dynamical_field(const Parameters &parameters, const std::vector<dcomp> &self_energy_retarded)
{
    for (int r = 0; r < parameters.steps_myid; r++){
        this->dynamical_field_retarded.at(r) = 1.0 / (1.0 / this->impurity_gf_mb_retarded.at(r) + self_energy_retarded.at(r));
        //std::cout << this->dynamical_field_retarded.at(r) << " " << this->impurity_gf_mb_retarded.at(r) << " " <<  self_energy_retarded.at(r) <<  " " << r << std::endl;
    }
    //write_to_file(parameters, dynamical_field_retarded, "dynamical_field_retarded", 0);
}


void AIM::get_lesser_hybridisation(const Parameters &parameters, const std::vector<dcomp> &self_energy_lesser)
{  
    for (int r = 0; r < parameters.steps_myid; r++){
        this->hybridisation_lesser.at(r) = ((1.0 / this->impurity_gf_mb_retarded.at(r)) * parameters.j1 * this->impurity_gf_mb_lesser.at(r) *
                                           1.0 / std::conj(this->impurity_gf_mb_retarded.at(r)) - self_energy_lesser.at(r)).imag();
    }
}       


void AIM::get_lesser_greater_hybridisation(const Parameters &parameters, const std::vector<dcomp> &self_energy_lesser, const std::vector<dcomp> &self_energy_greater)
{  
    for (int r = 0; r < parameters.steps_myid; r++){
        //std::cout << self_energy_lesser.at(r) << " " << self_energy_greater.at(r) << "\n";
        this->hybridisation_lesser.at(r) = ((1.0 / this->impurity_gf_mb_retarded.at(r)) * parameters.j1 * this->impurity_gf_mb_lesser.at(r) *
                                           1.0 / std::conj(this->impurity_gf_mb_retarded.at(r)) - self_energy_lesser.at(r)).imag();
        this->hybridisation_greater.at(r) = ((1.0 / this->impurity_gf_mb_retarded.at(r)) * parameters.j1 * this->impurity_gf_mb_greater.at(r) *
                                           1.0 / std::conj(this->impurity_gf_mb_retarded.at(r)) - self_energy_greater.at(r)).imag();
        //std::cout << this->hybridisation_greater.at(r) << " " << hybridisation_lesser.at(r) << "\n";
        //std::cout << this->impurity_gf_mb_lesser.at(r) << " " << this->impurity_gf_mb_greater.at(r) << " " << (1.0 / this->impurity_gf_mb_retarded.at(r)) * 
        //1.0 / std::conj(this->impurity_gf_mb_retarded.at(r)) << " " <<  self_energy_lesser.at(r) << "\n";
    }


    //write_to_file(parameters, impurity_gf_mb_lesser, "impurity_gf_mb_lesser.dat", 0);
    //write_to_file(parameters, impurity_gf_mb_greater, "impurity_gf_mb_greater.dat", 0);
    //write_to_file(parameters, self_energy_lesser, "self_energy_lesser.dat", 0);
    //write_to_file(parameters, self_energy_greater, "self_energy_greater.dat", 0);
    //write_to_file(parameters, hybridisation_lesser, "hydrisation_lesser.dat", 0);
    //write_to_file(parameters, hybridisation_greater, "hydrisation_greater.dat", 0);
}    

void AIM::get_dynamical_field_lesser(const Parameters &parameters, const int voltage_step)
{
    for (int r = 0; r < parameters.steps_myid; r++){
        dcomp advanced = std::conj(this->dynamical_field_retarded.at(r));
        this->dynamical_field_lesser.at(r) = (this->dynamical_field_retarded.at(r) * this->hybridisation_lesser.at(r) * advanced).real(); //need to be sure that this is indeed imaginary
    }
}

void AIM::get_dynamical_field_lesser_greater(const Parameters &parameters, const int voltage_step)
{
    for (int r = 0; r < parameters.steps_myid; r++){
        dcomp advanced = std::conj(this->dynamical_field_retarded.at(r));
        this->dynamical_field_lesser.at(r) = (this->dynamical_field_retarded.at(r) * this->hybridisation_lesser.at(r) * advanced).real(); //need to be sure that this is indeed imaginary
        this->dynamical_field_greater.at(r) = (this->dynamical_field_retarded.at(r) * this->hybridisation_greater.at(r) * advanced).real(); //need to be sure that this is indeed imaginary
    }
}

void AIM::get_effective_fermi_function(const Parameters &parameters, const int voltage_step){
    for (int r = 0; r < parameters.steps_myid; r++){
        dcomp advanced = std::conj(this->dynamical_field_retarded.at(r));
        this->fermi_function_eff.at(r) = (this->dynamical_field_lesser.at(r) / (this->dynamical_field_retarded.at(r) - advanced)).imag();
    }
    //write_to_file(parameters, this->fermi_function_eff, "fermi_effective", voltage_step);
}

