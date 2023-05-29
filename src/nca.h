#include "parameters.h"
#include "pseudo_gf.h"
#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
#include <limits>

void impurity_solver_nca(const Parameters &parameters, const int voltage_step, AIM &aim_up, AIM &aim_down);

void impurity_solver(const Parameters &parameters, Interacting_GF &boson, Interacting_GF &fermion_up, Interacting_GF &fermion_down,
     int voltage_step, double &z_prefactor);

void get_greater_lesser_se_fermion(const Parameters &parameters, const Interacting_GF &boson, Interacting_GF &fermion, int voltage_step);

void get_retarded_se(const Parameters &parameters, Interacting_GF &green_function);

void get_retarded_gf_fermion(const Parameters &parameters, Interacting_GF &green_function);

void get_greater_lesser_gf(const Parameters &parameters, Interacting_GF &green_function);

void get_greater_lesser_gf(const Parameters &parameters, Interacting_GF &green_function, std::vector<double> &old_gf_lesser, std::vector<double> &old_gf_greater);

void get_greater_lesser_se_boson(const Parameters &parameters, Interacting_GF &boson, const Interacting_GF &fermion_up,
    const Interacting_GF &fermion_down, int voltage_step);

void get_retarded_gf_boson(const Parameters &parameters, Interacting_GF &green_function);

void get_difference(const Parameters &parameters, std::vector<dcomp> &old_self_energy, std::vector<dcomp> &new_self_energy,
                double &difference);

double absolute_value(double num1);

double get_z_prefactor(const Parameters &parameters, Interacting_GF &boson, Interacting_GF &fermion_up, Interacting_GF &fermion_down);


void test_retarded_gf(const Parameters &parameters, Interacting_GF &boson, Interacting_GF &fermion_up, Interacting_GF &fermion_down);

