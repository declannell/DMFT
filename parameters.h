#pragma once
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <math.h>

typedef std::complex<double> dcomp;


struct Parameters
{
    double onsite;         // onsite energy in the scattering region
    double onsite_l;       // onsite energy in the left lead
    double onsite_r;       // onsite energy in the right lead
    double hopping;        // the hopping the z direction of the scattering region
    double hopping_y;      // the double hopping the y direction of the scattering region
    double hopping_x;      // the hopping in the x direction of the scattering region
    double hopping_lz;     // the hopping in the z direction of the left lead
    double hopping_ly;     // the hopping in the y direction of the left lead
    double hopping_lx;     // the hopping in the x direction of the left lead
    double hopping_rz;     // the hopping in the z direction of the right lead
    double hopping_ry;     // the hopping in the y direction of the right lead
    double hopping_rx;     // the hopping in the x direction of the left lead
    double hopping_lc;     // the hopping inbetween the left lead and scattering region
    double hopping_rc;     // the hopping inbetween the right lead and scattering region
    double chain_length;   // the number of atoms in the z direction of the scattering region
    double chain_length_y; // this is the number of k in the y direction for the scattering region
    double chain_length_x; // This is the number of points in the x direction.
    double chemical_potential;
    double temperature;
    double e_upper_bound;       // this is the max energy value
    double e_lower_bound;       // this is the min energy value
    double hubbard_interaction; // this is the hubbard interaction
    int voltage_step;        // voltage step of zero is equilibrium. This is an integer and higher values correspond to a higher potential difference between the two leads.
    double pi;
    double self_consistent_steps; // this is the number of self consistent steps my code needs
    bool read_in_self_energy;
    int NIV_points;
    double delta_v;
    int interaction_order; // this is the order the green function will be calculated too in terms of interaction strength. this can be equal to 0 , 1 or 2//
    std::string path_of_self_energy_up;
    std::string path_of_self_energy_down;
    std::vector<double> voltage_r;
    std::vector<double> voltage_l;
    int steps; // number of energy points we take
    std::vector<dcomp> energy;
    static Parameters init();
    dcomp j1; // this is a complex number class defined within the complex library
};

# define M_PI           3.14159265358979323846

double fermi_function(double energy, const Parameters &parameters);