#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include "parameters.h"
#include "leads_self_energy.h"
#include "interacting_gf.h"
#include "dmft.h"
#include "transport.h"
#include "analytic_gf.h"
#include "aim.h"
#include <mpi.h>


void decomp(int steps, int size, int myid, int *s, int *e);

void get_momentum_vectors(std::vector<double> &kx, std::vector<double> &ky, Parameters &parameters);

void write_to_file(const Parameters &parameters, std::vector<double> &gf_up, std::string filename, int voltage_step);

void write_to_file(const Parameters &parameters, std::vector<dcomp> &gf_up, std::string filename, int voltage_step);

void write_to_file(const Parameters &parameters, MatrixVectorType &gf_up, MatrixVectorType &gf_down, std::string filename, int voltage_step);

void write_to_file(const Parameters &parameters, std::vector<dcomp> &gf_up, std::vector<dcomp> &gf_down, std::string filename, int voltage_step);

void write_to_file(const Parameters &parameters, std::vector<std::vector<dcomp>> &se_up, std::vector<std::vector<dcomp>> &se_down, std::string filename, int voltage_step);

void distribute_to_procs(const Parameters &parameters, std::vector<dcomp> &vec_1, std::vector<dcomp> &vec_2);

void distribute_to_procs(const Parameters &parameters, std::vector<double> &vec_1, const std::vector<double> &vec_2);

double kramer_kronig_relation(const Parameters &parameters, std::vector<double> &impurity_self_energy_imag, int r);

double absolute_value(double num1);

void integrate_spectral(Parameters &parameters, MatrixVectorType &gf_local);

void get_occupation(Parameters  &parameters, MatrixVectorType & gf_local_lesser_up, 
	MatrixVectorType & gf_local_lesser_down, std::vector<double> &spins_occup);

void get_dos(Parameters &parameters, std::vector<dcomp> &dos_up, std::vector<dcomp> &dos_down, std::vector<dcomp> &dos_up_ins, std::vector<dcomp> &dos_down_ins,
 	std::vector<dcomp> &dos_up_metal, std::vector<dcomp> &dos_down_metal, MatrixVectorType &gf_local_up, MatrixVectorType &gf_local_down);



