#include "parameters.h"
#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
//#include <eigen3/Eigen/Dense>
#include <limits>
#include "pseudo_gf.h"
#include "utilis.h"


Pseudo_GF::Pseudo_GF(const Parameters &parameters)
{
	this->greater_gf.resize(parameters.steps_myid, 0), this->lesser_gf.resize(parameters.steps_myid, 0);
    this->retarded_gf.resize(parameters.steps_myid, 0);
    this->greater_se.resize(parameters.steps_myid, 0), this->lesser_se.resize(parameters.steps_myid, 0), this->retarded_se.resize(parameters.steps_myid, 0);   

    //std::cout << greater_gf.size() << std::endl;
}

void Pseudo_GF::print_green_function(const Parameters &parameters, int voltage_step, const std::string& name) {

    std::ostringstream ossgf;
	ossgf << name <<  "_greater_gf.dat";
	std::string var = ossgf.str();
	std::ofstream gf_greater_file;
	write_to_file(parameters, this->greater_gf, var, voltage_step);

    ossgf.str("");
    ossgf.clear();
	ossgf << name <<  "_lesser_gf.dat";
	var = ossgf.str();
	write_to_file(parameters, this->lesser_gf, var, voltage_step);

    ossgf.str("");
    ossgf.clear();
	ossgf << name <<  "_retarded_gf.dat";
	var = ossgf.str();
	write_to_file(parameters, this->retarded_gf, var, voltage_step);

    ossgf.str("");
    ossgf.clear();
	ossgf << name <<  "_greater_se.dat";
	var = ossgf.str();
	write_to_file(parameters, this->greater_se, var, voltage_step);

	ossgf.str("");
	ossgf.clear();
	ossgf << name <<  "_lesser_se.dat";
	var = ossgf.str();
	write_to_file(parameters, this->lesser_se, var, voltage_step);

   	ossgf.str("");
   	ossgf.clear();
   	ossgf << name <<  "_retarded_se.dat";
	var = ossgf.str();
	write_to_file(parameters, this->retarded_se, var, voltage_step);
}	