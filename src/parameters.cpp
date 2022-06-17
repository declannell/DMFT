#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include "parameters.h"
#include <iostream>

Parameters Parameters::init()
{
    Parameters parameters =  {
        .onsite_cor = -0.0,
        .onsite_ins_l = -0.18,
        .onsite_ins_r = -0.18,
        .onsite_l = -0.0,
        .onsite_r = -0.0,
        .hopping_cor = -0.10,
        .hopping_ins_l = -0.10,
        .hopping_ins_r = -0.10,
        .hopping_y = -0.10,
        .hopping_x = -0.10,
        .hopping_lz = -0.10,
        .hopping_ly = -0.10,
        .hopping_lx = -0.10,
        .hopping_rz = -0.10,
        .hopping_ry = -0.10,
        .hopping_rx = -0.10,
        .hopping_lc = -0.10,
        .hopping_rc = -0.10,
        .hopping_ins_l_cor = -0.1,
        .hopping_ins_r_cor = -0.1,        
        .num_cor = 1, //this is the number of correlated atoms between the insulating atoms.
        .num_ins_left  = 2, //this is the number of insulating layers on the left side.    
        .num_ins_right = 2,
        .num_ky_points = 400,
        .num_kx_points = 400,
        .chemical_potential = 0.0,
        .temperature = 00.0,
        .e_upper_bound = 4.0,
        .e_lower_bound = -4.0,
        .hubbard_interaction = 0.03,
        .voltage_step = 0,
        .self_consistent_steps = 45,
        .read_in_self_energy = false,
        .NIV_points = 10,
        .delta_v = 0.008,
        .delta_leads = 0.000000001,
        .delta_gf = 0.000001,
        .leads_3d = false
    
    };

    parameters.path_of_self_energy_up = "/home/declan/green_function_code/quantum_transport/c++/DMFT/textfiles/local_se_up_1_k_points_81_energy.txt";
    parameters.path_of_self_energy_down = "/home/declan/green_function_code/quantum_transport/c++/DMFT/textfiles/local_se_down_1_k_points_81_energy.txt";

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

    parameters.steps = 901; //you must make sure the energy spacing is less than delta_v
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
