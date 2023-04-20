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
#include "AIM.h"
#include <mpi.h>


void decomp(int steps, int size, int myid, int *s, int *e);

void get_momentum_vectors(std::vector<double> &kx, std::vector<double> &ky, Parameters &parameters);

void write_to_file(const Parameters &parameters, std::vector<double> &gf_up, std::string filename, int voltage_step);

void write_to_file(Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_up, std::vector<Eigen::MatrixXcd> &gf_down, std::string filename, int voltage_step);

void write_to_file(Parameters &parameters, std::vector<dcomp> &gf_up, std::vector<dcomp> &gf_down, std::string filename, int voltage_step);

void write_to_file(Parameters &parameters, std::vector<std::vector<dcomp>> &se_up, std::vector<std::vector<dcomp>> &se_down, std::string filename, int voltage_step);

void distribute_to_procs(const Parameters &parameters, std::vector<dcomp> &vec_1, std::vector<dcomp> &vec_2);

void distribute_to_procs(const Parameters &parameters, std::vector<double> &vec_1, std::vector<double> &vec_2);

