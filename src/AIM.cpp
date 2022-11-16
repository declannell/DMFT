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
#include "AIM.h"
#include <limits>

AIM::AIM(const Parameters &parameters, const std::vector<dcomp> &local_gf_retarded, const std::vector<dcomp> &local_gf_lesser, 
    const std::vector<dcomp> &self_energy_retarded, const std::vector<dcomp> &self_energy_lesser, const int voltage_step)
{
    this->impurity_gf_mb_retarded.resize(parameters.steps_myid);
    this->dynamical_field_retarded.resize(parameters.steps_myid);
    this->impurity_gf_mb_lesser.resize(parameters.steps_myid);
    this->dynamical_field_lesser.resize(parameters.steps_myid);
    this->fermi_function_eff.resize(parameters.steps_myid);
    this->self_energy_mb_retarded.resize(parameters.steps_myid);
    this->self_energy_mb_lesser.resize(parameters.steps_myid);
    this->hybridisation_lesser.resize(parameters.steps_myid);

    get_impurity_gf_mb(parameters, local_gf_retarded, local_gf_lesser);
    get_retarded_dynamical_field(parameters, local_gf_retarded, self_energy_retarded);
    get_lesser_hybridisation(parameters, self_energy_lesser);   
    get_dynamical_field_lesser(parameters, voltage_step);
    get_effective_fermi_function(parameters); 
}

void AIM::get_impurity_gf_mb(const Parameters &parameters, const std::vector<dcomp> &local_gf_retarded, const std::vector<dcomp> &local_gf_lesser)
{
    //std::ostringstream ossgf;
	//ossgf << "textfiles/" << 1 << "."  << ".mb_imp_gf.txt";
	//std::string var = ossgf.str();
//
	//std::ofstream mb_imp_gf;
    //mb_imp_gf.open(var);

    for (int r = 0; r < parameters.steps_myid; r++){
        this->impurity_gf_mb_retarded.at(r) = local_gf_retarded.at(r);
        this->impurity_gf_mb_lesser.at(r) = local_gf_lesser.at(r).imag();
        //mb_imp_gf << parameters.energy.at(r) << "  " << this->impurity_gf_mb_retarded.at(r).real() << "  " << this->impurity_gf_mb_retarded.at(r).imag()
        // << "  " << this->impurity_gf_mb_lesser.at(r) << " \n";
    }

	//mb_imp_gf.close();

}

void AIM::get_retarded_dynamical_field(const Parameters &parameters, const std::vector<dcomp> &local_gf_retarded, const std::vector<dcomp> &self_energy_retarded)
{
    //std::ostringstream ossgf;
	//ossgf << "textfiles/" << 1 << "."  << ".dynamical_field_retarded.txt";
	//std::string var = ossgf.str();
//
	//std::ofstream mb_imp_gf;
    //mb_imp_gf.open(var);

    for (int r = 0; r < parameters.steps_myid; r++){
        this->dynamical_field_retarded.at(r) = 1.0 / (1.0 / local_gf_retarded.at(r) + self_energy_retarded.at(r));
        //mb_imp_gf << parameters.energy.at(r) << "  " << this->dynamical_field_retarded.at(r).real() << "  " << this->dynamical_field_retarded.at(r).imag() << " " 
        //        << self_energy_retarded.at(r) << " \n";
    }

	//mb_imp_gf.close();
}

void AIM::get_lesser_hybridisation(const Parameters &parameters, const std::vector<dcomp> &self_energy_lesser)
{  
    //std::ostringstream ossgf;
	//ossgf << "textfiles/" << 1 << "."  << ".hybridisation_lesser.txt";
	//std::string var = ossgf.str();
//
	//std::ofstream hybridisation_lesser;
    //hybridisation_lesser.open(var);

    for (int r = 0; r < parameters.steps_myid; r++){
        this->hybridisation_lesser.at(r) = ((1.0 / this->impurity_gf_mb_retarded.at(r)) * parameters.j1 * this->impurity_gf_mb_lesser.at(r) *
                                           std::conj(1.0 / this->impurity_gf_mb_retarded.at(r)) - self_energy_lesser.at(r)).imag();
        //this->hybridisation_lesser.at(r) = ((1.0 / this->impurity_gf_mb_retarded.at(r)) * this->impurity_gf_mb_lesser.at(r) * 
        //                    std::conj((1.0 / this->impurity_gf_mb_retarded.at(r))) - this->impurity_gf_mb_lesser.at(r)).real();
        //this->hybridisation_lesser.at(r) =  - (this->fermi_function_eff.at(r) * (2.0 * parameters.j1 * parameters.delta_gf +
        //                                    (1.0 / advanced) - 1.0 / (this->dynamical_field_retarded.at(r)))).imag();

        //hybridisation_lesser << parameters.energy.at(r) << "  " << this->hybridisation_lesser.at(r)  << " \n";
    }
	//hybridisation_lesser.close();

}                        

void AIM::get_dynamical_field_lesser(const Parameters &parameters, const int voltage_step)
{
    for (int r = 0; r < parameters.steps_myid; r++){
        dcomp advanced = std::conj(this->dynamical_field_retarded.at(r));
        this->dynamical_field_lesser.at(r) = (this->dynamical_field_retarded.at(r) * this->hybridisation_lesser.at(r) * advanced).real(); //need to be sure that this is indeed imaginary
    }

    //std::ostringstream ossgf;
	//ossgf << "textfiles/" << 1 << "."  << ".dynamical_field_lesser.txt";
	//std::string var = ossgf.str();
	//std::ofstream mb_imp_gf;
    //mb_imp_gf.open(var);
//
    //for (int r = 0; r < parameters.steps; r++){
    //    mb_imp_gf << parameters.energy.at(r) << "  " << this->dynamical_field_lesser.at(r)  << " \n";
    //}
//
	//mb_imp_gf.close();

}


void AIM::get_effective_fermi_function(const Parameters &parameters){

    //std::ostringstream ossgf;
	//ossgf << "textfiles/" << 1 << "."  << ".effective_fermi.txt";
	//std::string var = ossgf.str();
//
	//std::ofstream effective_fermi_function;
    //effective_fermi_function.open(var);

    for (int r = 0; r < parameters.steps_myid; r++){
        dcomp advanced = std::conj(this->dynamical_field_retarded.at(r));
        this->fermi_function_eff.at(r) = (this->dynamical_field_lesser.at(r) / (this->dynamical_field_retarded.at(r) - advanced)).imag();

        //dcomp fermi_hybridisation = this->hybridisation_lesser.at(r) / (2.0 * parameters.j1 * parameters.delta_gf +
        //                                    (1.0 / advanced) - 1.0 / (this->dynamical_field_retarded.at(r)));

        //std::cout << r << " " << fermi_function_eff.at(r) << std::endl;

        //effective_fermi_function << parameters.energy.at(r) << "  " << this->fermi_function_eff.at(r) << " " << fermi_hybridisation.imag() << "  " 
        //        << this->hybridisation_lesser.at(r) << "    " <<  (2.0 * parameters.j1 * parameters.delta_gf +
        //                                    (1.0 / advanced) - 1.0 / (this->dynamical_field_retarded.at(r))) << " \n";
    }
	//effective_fermi_function.close();
}

