#include "parameters.h"
#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
//#include <eigen3/Eigen/Dense>
#include <limits>


Pseudo_GF::Pseudo_GF(const Parameters &parameters)
{
	 this->greater_gf.resize(parameters.steps, 0), this->lesser_gf.resize(parameters.steps, 0);
    this->retarded_gf.resize(parameters.steps, 0);
    this->greater_se.resize(parameters.steps, 0), this->lesser_se.resize(parameters.steps, 0), this->retarded_se.resize(parameters.steps, 0);   

    //std::cout << greater_gf.size() << std::endl;
}

void Pseudo_GF::print_green_function(const Parameters &parameters, int voltage_step, const std::string& name) {
    std::ostringstream ossgf;
	ossgf << voltage_step << "." << name <<  "_greater_gf.dat";
	std::string var = ossgf.str();
	std::ofstream gf_greater_file;
	gf_greater_file.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		gf_greater_file << parameters.energy.at(r) << "  " << this->greater_gf.at(r).real() << "   " << this->greater_gf.at(r).imag() << "  \n";
	}
	gf_greater_file.close();

    ossgf.str("");
    ossgf.clear();
	ossgf << voltage_step << "." << name <<  "_lesser_gf.dat";
	var = ossgf.str();
	std::ofstream gf_lesser_file;
	gf_lesser_file.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		gf_lesser_file << parameters.energy.at(r) << "  " << this->lesser_gf.at(r).real() << "   " << this->lesser_gf.at(r).imag() << "  \n";
	}
	gf_lesser_file.close();

    ossgf.str("");
    ossgf.clear();
	ossgf << voltage_step << "." << name <<  "_retarded_gf.dat";
	var = ossgf.str();
	std::ofstream gf_retarded_file;
	gf_retarded_file.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		gf_retarded_file << parameters.energy.at(r) << "  " << this->retarded_gf.at(r).real() << "   " << this->retarded_gf.at(r).imag() << "  \n";
	}
	gf_retarded_file.close();

    ossgf.str("");
    ossgf.clear();
	ossgf << voltage_step << "." << name <<  "_greater_se.dat";
	var = ossgf.str();
	std::ofstream se_greater_file;
	se_greater_file.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		se_greater_file << parameters.energy.at(r) << "  " << this->greater_se.at(r).real() << "   " << this->greater_se.at(r).imag() << "  \n";
	}
	se_greater_file.close();

    ossgf.str("");
    ossgf.clear();
	ossgf << voltage_step << "." << name <<  "_lesser_se.dat";
	var = ossgf.str();
	std::ofstream se_lesser_file;
	se_lesser_file.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		se_lesser_file << parameters.energy.at(r) << "  " << this->lesser_se.at(r).real() << "   " << this->lesser_se.at(r).imag() << "  \n";
	}
	se_lesser_file.close();

    ossgf.str("");
    ossgf.clear();
    ossgf << voltage_step << "." << name <<  "_retarded_se.dat";
	var = ossgf.str();
	std::ofstream se_retarded_file;
	se_retarded_file.open(var);
	for (int r = 0; r < parameters.steps; r++) {
		se_retarded_file << parameters.energy.at(r) << "  " << this->retarded_se.at(r).real() << "   " << this->retarded_se.at(r).imag() << "  \n";
	}
	se_retarded_file.close();
}