#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include "parameters.h"
#include <iostream>

Parameters Parameters::init()
{
    Parameters parameters =  {
        .onsite_cor = -.035,
        .onsite_ins_l = 10,
        .onsite_ins_r = 10,
        .onsite_l = -0.0,
        .onsite_r = -0.0,
        .hopping_cor = -2.5,
        .hopping_ins_l = -2.5,
        .hopping_ins_r = -2.5,
        .hopping_y = -2.5,
        .hopping_x = -2.5,
        .hopping_lz = -2.5,
        .hopping_ly = -2.5,
        .hopping_lx = -2.5,
        .hopping_rz = -2.5,
        .hopping_ry = -2.5,
        .hopping_rx = -2.5,
        .hopping_lc = -2.5,
        .hopping_rc = -2.5,
        .hopping_ins_l_cor = -2.5,
        .hopping_ins_r_cor = -2.5,        
        .num_cor = 1, //this is the number of correlated atoms between the insulating atoms.
        .num_ins_left  = 0, //this is the number of insulating layers on the left side.    
        .num_ins_right = 0,
        .num_ky_points = 1,
        .num_kx_points = 1,
        .chemical_potential = 0.0,
        .temperature = 00.0,
        .e_upper_bound = 40.0,
        .e_lower_bound = -40.0,
        .hubbard_interaction = 2.5,
        .voltage_step = 0,
        .self_consistent_steps = 30,
        .read_in_self_energy = false,
        .NIV_points = 6,
        .delta_v = 0.5,
        .delta_leads = 0.00000001,
        .delta_gf = 0.0000000001,
        .leads_3d = false,
        .spin_up_occup = 0.0,
        .spin_down_occup = 0.0,       
        .convergence = 0.0001 
    
    };

    parameters.path_of_self_energy_up = "textfiles/local_se_up_1_k_points_81_energy.txt";
    parameters.path_of_self_energy_down = "textfiles/local_se_down_1_k_points_81_energy.txt";

    parameters.voltage_l.resize(parameters.NIV_points);
    parameters.voltage_r.resize(parameters.NIV_points);
    for (int i = 0; i < parameters.NIV_points; i++)
    {
        parameters.voltage_l.at(i) = parameters.delta_v * (double)(i);
        parameters.voltage_r.at(i) = - parameters.delta_v * (double)(i);
    }

    if (parameters.hubbard_interaction == 0.0)
    {
        parameters.interaction_order = 0.0; // this is the order the green function will be calculated too in terms of interaction strength. this can be equal to 0 , 1 or 2//
    }
    else
    {
        parameters.interaction_order = 2;
    }

    parameters.chain_length = parameters.num_ins_left + parameters.num_ins_right + parameters.num_cor;

    parameters.steps = 401; //you must make sure the energy spacing is less than delta_v
    parameters.energy.resize(parameters.steps);

    parameters.j1 = -1;
    parameters.j1 = sqrt(parameters.j1);

    double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;

    for (int i = 0; i < parameters.steps; i++)
    {
        parameters.energy.at(i) = parameters.e_lower_bound + delta_energy * (double)i;
    }
    return parameters;

    if(delta_energy < parameters.delta_v){
        std::cout << "Delta voltage is less than delta energy. This gives unphysical step function results. Make delta_energy < parameters.delta_v" <<std::endl;
    }
}

double fermi_function(double energy, const Parameters &parameters) {
    if(parameters.temperature == 0){
        if(energy < parameters.chemical_potential){
            return 1.0;
        } else {
            return 0.0;
        }
    } else {}
        return 1.0 / (1.0 + exp((energy - parameters.chemical_potential) / parameters.temperature));
}
//Parameters params = Parameters::init();
