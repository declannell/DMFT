#pragma once
#include <complex>
#include <cmath>
#include <vector>

typedef std::complex<double> dcomp;


struct Parameters
{
    double onsite_cor;         // onsite energy in the correlated region
    double onsite_ins_l;   // onsite energy in the left insulating region
    double onsite_ins_r;   // onsite energy in the right insulating region
    double onsite_l;       // onsite energy in the left lead
    double onsite_r;       // onsite energy in the right lead
    double hopping_cor;        // the hopping the z direction of the scattering region
    double hopping_ins_l; //the hopping in the left insulating region
    double hopping_ins_r; //the hopping in the right insulating region    
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
    double hopping_ins_l_cor; //the hopping between the insulating layers on the left and the correlated region    
    double hopping_ins_r_cor; //the hopping between the insulating layers on the right and the correlated region
    int num_cor; //this is the number of correlated atoms between the insulating atoms.
    int num_ins_left; //this is the number of insulating layers on the left side.    
    int num_ins_right; //this is the number of insulating layers on the right side.
    bool ins_metal_ins;//if false, the arrangment will be metal_ins_metal and the values for the hamiltonian of the insulating sites are set by ins_l. ins_r are never called.
    //if true the arrangement is ins_metal_ins and both ins_l and ins_r have an effect.
    int num_ky_points; // this is the number of k in the y direction for the scattering region
    int num_kx_points; // This is the number of points in the x direction.
    double chemical_potential;
    double temperature;
    double e_upper_bound;       // this is the max energy value
    double e_lower_bound;       // this is the min energy value
    double hubbard_interaction; // this is the hubbard interaction
    int voltage_step;        // voltage step of zero is equilibrium. This is an integer and higher values correspond to a higher potential difference between the two leads.
    double self_consistent_steps; // this is the number of self consistent steps my code needs
    bool read_in_self_energy;
    int NIV_points;//number of IV points
    double delta_v; //the voltage step between IV points
    double delta_leads; //delta in the leads
    double delta_gf; //delta in the scattering region
    bool leads_3d;//if true, this will attach 3d leads to a 1d scattering region
    double spin_up_occup;
    double spin_down_occup;
    double convergence;
    int chain_length;   // the number of atoms in the z direction of the scattering region
    std::vector<int> atom_type; //this is a list of whether the atom is an insulator (=0) or a metal (= 1)
    int interaction_order; // this is the order the green function will be calculated too in terms of interaction strength. this can be equal to 0 , 1 or 2//
    std::string path_of_self_energy_up;
    std::string path_of_self_energy_down;
    std::vector<double> voltage_r;
    std::vector<double> voltage_l;
    int steps; // number of energy points we take
    std::vector<double> energy;
    static Parameters init();
    dcomp j1; // this is a complex number class defined within the complex library
};



double fermi_function(double energy, const Parameters &parameters);
