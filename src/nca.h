#pragma once
#include "parameters.h"
#include "pseudo_gf.h"
#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
#include <limits>
#include "parameters.h"
#include "pseudo_gf.h"
#include "nca.h"
#include "utilis.h"
#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
#include <limits>


void get_lesser_greater_gf(const Parameters &parameters, const Pseudo_GF &boson, const Pseudo_GF &fermion, 
     const double &z_prefactor, AIM &aim_up);

void get_difference_self_energy(const Parameters &parameters, std::vector<dcomp> &self_energy_mb_up,
	 std::vector<dcomp> &old_self_energy_mb_up, double &difference, int &index);

void intialise_gf(const Parameters &parameters, Pseudo_GF &boson, Pseudo_GF &fermion_up, Pseudo_GF &fermion_down);

void get_greater_lesser_se_fermion(const Parameters &parameters, const Pseudo_GF &boson, Pseudo_GF &fermion, int voltage_step, AIM &aim_up);


void get_retarded_gf(const Parameters &parameters, Pseudo_GF &green_function);

void get_greater_lesser_gf(const Parameters &parameters, Pseudo_GF &green_function);

void get_greater_lesser_se_boson(const Parameters &parameters, Pseudo_GF &boson, const Pseudo_GF &fermion_up,
    const Pseudo_GF &fermion_down, int voltage_step, AIM &aim_up, AIM &aim_down);

void test_retarded_gf(const Parameters &parameters, Pseudo_GF &boson, Pseudo_GF &fermion_up, Pseudo_GF &fermion_down);

void get_greater_lesser_gf(const Parameters &parameters, Pseudo_GF &green_function, int count);

double get_z_prefactor(const Parameters &parameters, Pseudo_GF &boson, Pseudo_GF &fermion_up, Pseudo_GF &fermion_down);

void solve_pseudo_particle_gf(const Parameters &parameters, Pseudo_GF &boson, Pseudo_GF &fermion_up, Pseudo_GF &fermion_down, int voltage_step,
     double &z_prefactor, AIM &aim_up, AIM &aim_down);

void get_retarded_impurity_se(const Parameters &parameters, double z_prefactor, AIM &aim_up);

void get_lesser_greater_impurity_se(const Parameters &parameters, AIM &aim_up);

void integrate_dos(const Parameters &parameters, Pseudo_GF &fermion_up, double &z_prefactor, AIM &aim_up);

void impurity_solver_nca(const Parameters &parameters, const int voltage_step, AIM &aim_up, AIM &aim_down);


