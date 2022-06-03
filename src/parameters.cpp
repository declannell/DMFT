#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include "parameters.h"
#include <iostream>

Parameters Parameters::init()
{
    Parameters parameters =  {
        .onsite = -0.15,
        .onsite_l = -0.0,
        .onsite_r = -0.0,
        .hopping = -1.0,
        .hopping_y = -1.0,
        .hopping_x = -1.0,
        .hopping_lz = -1.0,
        .hopping_ly = -1.0,
        .hopping_lx = -1.0,
        .hopping_rz = -1.0,
        .hopping_ry = -1.0,
        .hopping_rx = -1.0,
        .hopping_lc = -1.0,
        .hopping_rc = -1.0,
        .chain_length = 1,
        .chain_length_y = 1,
        .chain_length_x = 1,
        .chemical_potential = 0.0,
        .temperature = 10.0,
        .e_upper_bound = 15.0,
        .e_lower_bound = -15.0,
        .hubbard_interaction = 0.3,
        .voltage_step = 0,
        .self_consistent_steps = 20,
        .read_in_self_energy = false,
        .NIV_points = 8,
        .delta_v = 0.05,
        .delta_leads = 0.000000001,
        .delta_gf = 0.0000000001,
    
    };

    parameters.path_of_self_energy_up = "/home/declan/green_function_code/quantum_transport/textfiles/local_se_up_1_k_points_81_energy.txt";
    parameters.path_of_self_energy_down = "/home/declan/green_function_code/quantum_transport/textfiles/local_se_down_1_k_points_81_energy.txt";

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

    parameters.steps = 401; //you must make sure the energy spacing is less than delta_v
    parameters.energy.resize(parameters.steps);

    parameters.j1 = -1;
    parameters.j1 = sqrt(parameters.j1);
    std::cout << "The imaginary number is i is " << parameters.j1 << "\n";

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