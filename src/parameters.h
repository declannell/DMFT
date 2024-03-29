#pragma once
#include <complex>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <eigen3/Eigen/Dense>

typedef std::complex<double> dcomp;
using MatrixType = Eigen::MatrixXcd;
using MatrixVectorType = std::vector<MatrixType>;

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
    double self_consistent_steps; // this is the number of self consistent steps my code does
    double self_consistent_steps_nca; // this is the number of self consistent steps my code does for the nca loop
    double self_consistent_steps_leads; // this is the number of self consistent steps my code does for the leads.
    bool read_in_self_energy;
    int NIV_points;//number of IV points
	int NIV_start; //starting bias for the calculation. 0 for equilibrium
    double delta_v; //the voltage step between IV points
    double delta_leads; //delta in the leads
    double delta_gf; //delta in the scattering region
    bool leads_3d;//if true, this will attach 3d leads to a 1d scattering region
    double spin_up_occup;
    double spin_down_occup;
    double convergence;//the convergence of the mb self energy.
    double convergence_leads; //this is the convergence of the transfer matrix for the leads
    double gamma; //this is the value of the imag part of the self energy in the WBL.
    bool wbl_approx; //this sets whether or not we calculate the leads self energy by the wide band approx.
    //bool kk_relation; //try to integrate with the krammers kronig relation
    int chain_length;   // the number of atoms in the z direction of the scattering region. The number of atoms in the unit cell is 2 chain_length.
    std::vector<int> atom_type; //this is a list of whether the atom is an insulator (=0) or a metal (= 1)
    int interaction_order; // this is the order the green function will be calculated too in terms of interaction strength. this can be equal to 0 , 1 or 2//
    std::string path_of_self_energy_up;
    std::string path_of_self_energy_down;
    std::vector<double> voltage_r;
    std::vector<double> voltage_l;
    int steps; // number of energy points we take
    std::vector<double> energy;
    static Parameters from_file();
    dcomp j1; // this is a complex number class defined within the complex library
    int size; //the size of the communicator
    int myid; //the id of the process
    std::vector<int> start; //the starting index of the energy array for each process
    std::vector<int> end; //the ending index of the energy array for each process
    int steps_myid; //this is the number of steps the process has
    std::vector<int> steps_proc; //this is the number of steps the other processes have
    MPI_Comm comm;
    bool print_gf;
    bool spin_polarised; // 1 means we do a spin polarised calculations. 0 means a spin degenerate calculation.
    double delta_energy;
    int impurity_solver; //0 means mean field interaction, 1 means brute force sigma_2, 2 means kramer-kronig sigma_2, 3 means non crossing approx. 
    //4 means only second order with brute force. no mean field only. 5 means only second order with kramer-kronig. no mean field only
    double magnetic_field;//strenght of the magnetic field
    int half_metal; //if half metal == 1 then the insulating layers are replaced with half_metals.
    int meir_wingreen_current; // if meir_wingreen_current == 1, the current is calculated using the mw approach.
    int bond_current;  //if bond_current == 1, the current is calculated using the bond current approach
    std::vector<int> interface; // this defines where the interface for the bond currents is going to be. 
    int num_kx_leads_self_energy; //this is the number of k points for the leads self energy
    int num_ky_leads_self_energy; //this is the number of k points for the leads self energy
    int steps_leads; //this is the number of energy points for the leads. This will be 1 for the wideband approx as it is energy independent. 
};



double fermi_function(double energy, const Parameters &parameters);
void print_parameters(Parameters& parameters);
