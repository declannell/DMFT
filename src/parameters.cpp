#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include "parameters.h"
#include <iostream>

Parameters Parameters::init()
{
    Parameters parameters =  {
        .onsite_cor = 0.0,
        .onsite_ins_l = 10,
        .onsite_ins_r = 10,
        .onsite_l = -0.0,
        .onsite_r = -0.0,
        .hopping_cor = -1.0,
        .hopping_ins_l = -1.0,
        .hopping_ins_r = -1.0,
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
        .hopping_ins_l_cor = -1.0,
        .hopping_ins_r_cor = -1.0,
        .num_cor = 1, //this is the number of correlated atoms between the insulating atoms.
        .num_ins_left  = 0, //this is the number of insulating layers on the left side.    
        .num_ins_right = 0,
        .ins_metal_ins = true, 
        .num_ky_points = 1,
        .num_kx_points = 1,
        .chemical_potential = 0.0,
        .temperature = 00.0,
        .e_upper_bound = 20.0,
        .e_lower_bound = -20.0,
        .hubbard_interaction = 1,
        .voltage_step = 0,
        .self_consistent_steps = 1,
        .read_in_self_energy = false,
        .NIV_points = 2,
        .delta_v = 0.8,
        .delta_leads = 0.00000001,
        .delta_gf = 0.00,
        .leads_3d = false,
        .spin_up_occup = 0.0,
        .spin_down_occup = 0.0,       
        .convergence = 0.00001,
        .gamma = -0.5,
        .wbl_approx = true,
        .kk_relation = true
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
        parameters.interaction_order = 1;
    }
    if (parameters.ins_metal_ins == true){
        parameters.chain_length = parameters.num_ins_left + parameters.num_ins_right + parameters.num_cor;
    } else {
        parameters.chain_length = parameters.num_ins_left + 2 * parameters.num_cor;
    }

    //the atom is correlated if atom typ equals to 1. This how we know to apply sigma two to certain atoms.
    if (parameters.ins_metal_ins == true){
        for (int i = 0; i < parameters.num_ins_left; i++){
            parameters.atom_type.push_back(0);
        }
        for (int i = 0; i < parameters.num_cor; i++){
            parameters.atom_type.push_back(1);
        }
        for (int i = 0; i < parameters.num_ins_right; i++){
            parameters.atom_type.push_back(0);
        }
        //we repeat this a second time as there are two horizontal layers within the unit cell. atoms 0 to chain_length -1 are th e first layer.
        //the second layer is atoms chain_length to 2 * chain_length -1.
        for (int i = 0; i < parameters.num_ins_left; i++){
            parameters.atom_type.push_back(0);
        }
        for (int i = 0; i < parameters.num_cor; i++){
            parameters.atom_type.push_back(1);
        }
        for (int i = 0; i < parameters.num_ins_right; i++){
            parameters.atom_type.push_back(0);
        }
    } else {
        for (int i = 0; i < parameters.num_cor; i++){
            parameters.atom_type.push_back(1);
        }
        for (int i = 0; i < parameters.num_ins_left; i++){
            parameters.atom_type.push_back(0);
        }
        for (int i = 0; i < parameters.num_cor; i++){
            parameters.atom_type.push_back(1);
        }
        //repeating for the second layer.
        for (int i = 0; i < parameters.num_cor; i++){
            parameters.atom_type.push_back(1);
        }
        for (int i = 0; i < parameters.num_ins_left; i++){
            parameters.atom_type.push_back(0);
        }
        for (int i = 0; i < parameters.num_cor; i++){
            parameters.atom_type.push_back(1);
        }
    }

    for (int i = 0; i < 2 * parameters.chain_length; i++) {
        std::cout << parameters.atom_type.at(i) << std::endl;
    }

    parameters.steps = 200; //you must make sure the energy spacing is less than delta_v
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
void print_parameters(Parameters& parameters)
{
	std::cout << " .onsite_cor = " << parameters.onsite_cor << std::endl;
	std::cout << "onsite_ins_l = " << parameters.onsite_ins_l << std::endl;
	std::cout << "onsite_ins_r = " << parameters.onsite_ins_r << std::endl;
	std::cout << "onsite_l = " << parameters.onsite_l << std::endl;
	std::cout << "onsite_r = " << parameters.onsite_r << std::endl;
	std::cout << "hopping_cor = " << parameters.hopping_cor << std::endl;
	std::cout << "hopping_ins_l = " << parameters.hopping_ins_l << std::endl;
	std::cout << "hopping_ins_r = " << parameters.hopping_ins_r << std::endl;
	std::cout << "hopping_y = " << parameters.hopping_y << std::endl;
	std::cout << "hopping_x = " << parameters.hopping_x << std::endl;
	std::cout << "hopping_lz = " << parameters.hopping_lz << std::endl;
	std::cout << "hopping_ly = " << parameters.hopping_ly << std::endl;
	std::cout << "hopping_lx = " << parameters.hopping_lx << std::endl;
	std::cout << "hopping_rz = " << parameters.hopping_rz << std::endl;
	std::cout << "hopping_ry = " << parameters.hopping_ry << std::endl;
	std::cout << "hopping_rx = " << parameters.hopping_rx << std::endl;
	std::cout << "hopping_lc = " << parameters.hopping_lc << std::endl;
	std::cout << "hopping_rc = " << parameters.hopping_rc << std::endl;
	std::cout << "hopping_ins_l_cor = " << parameters.hopping_ins_l_cor << std::endl;
	std::cout << "hopping_ins_r_cor = " << parameters.hopping_ins_r_cor << std::endl;
	std::cout << "num_cor = " << parameters.num_cor << std::endl;
	std::cout << "parameters.num_ins_left  =" << parameters.num_ins_left << std::endl;
	std::cout << "num_ins_right = " << parameters.num_ins_right << std::endl;
	std::cout << "num_ky_points = " << parameters.num_ky_points << std::endl;
	std::cout << "num_kx_points = " << parameters.num_kx_points << std::endl;
	std::cout << "chemical_potential = " << parameters.chemical_potential << std::endl;
	std::cout << "temperature = " << parameters.temperature << std::endl;
	std::cout << "e_upper_bound = " << parameters.e_upper_bound << std::endl;
	std::cout << "e_lower_bound = " << parameters.e_lower_bound << std::endl;
	std::cout << "hubbard_interaction = " << parameters.hubbard_interaction << std::endl;
	std::cout << "voltage_step = " << parameters.voltage_step << std::endl;
	std::cout << "self_consistent_steps = " << parameters.self_consistent_steps << std::endl;
	std::cout << "read_in_self_energy = " << parameters.read_in_self_energy << std::endl;
	std::cout << "NIV_points = " << parameters.NIV_points << std::endl;
	std::cout << "delta_v = " << parameters.delta_v << std::endl;
	std::cout << "delta_leads = " << parameters.delta_leads << std::endl;
	std::cout << "delta_gf = " << parameters.delta_gf << std::endl;
	std::cout << "leads_3d = " << parameters.leads_3d << std::endl;
	std::cout << "parameters.interaction_order = " << parameters.interaction_order << std::endl;
	std::cout << "parameters.steps = " << parameters.steps << std::endl;
	std::cout << "parameters.chain_length = " << parameters.chain_length << std::endl;
	std::cout << "parameters.spin_up_occup = " << parameters.spin_up_occup << std::endl;
	std::cout << "parameters.spin_down_occup = " << parameters.spin_down_occup << std::endl;
}