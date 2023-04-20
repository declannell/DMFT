#include "parameters.h"

#include <complex>  //this contains complex numbers and trig functions
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void simple_tokenizer(std::string s, std::string &variable, std::string &value)
{
    std::stringstream ss(s);
    std::string word;
	int count = 0;
    while (ss >> word) {
        //std::cout << word << std::endl;
		if (count == 0) {
			variable = word;
		} else if (count == 1) {
			value = word;
		}
		count++;
    }
}


Parameters Parameters::from_file()
{
	Parameters parameters;
	std::string line, variable, value;
	std::ifstream input_file;

	input_file.open("input_file");

	if(!input_file.is_open())
	{
		std::cout << "The input_file doesn't exist \n" << std::endl;
		std::exit(1);
	} 
	
	while (getline(input_file, line)) {
			//std::cout << line << '\n';
			simple_tokenizer(line, variable, value);
			//std::cout << "The variable name is " << variable << " The value is  " << value << std::endl;
			if (variable == "onsite_cor") {
				parameters.onsite_cor = std::stod(value);
			} else if (variable == "onsite_ins_l") {
				parameters.onsite_ins_l = std::stod(value);
			} else if (variable == "onsite_ins_r") {
				parameters.onsite_ins_r = std::stod(value);
			} else if (variable == "onsite_l") {
				parameters.onsite_l = std::stod(value);
			} else if (variable == "onsite_r") {
				parameters.onsite_r = std::stod(value);
			} else if (variable == "hopping_cor") {
				parameters.hopping_cor = std::stod(value);
			} else if (variable == "hopping_ins_l") {
				parameters.hopping_ins_l = std::stod(value);
			} else if (variable == "hopping_ins_r") {
				parameters.hopping_ins_r = std::stod(value);
			} else if (variable == "hopping_y") {
				parameters.hopping_y = std::stod(value);
			} else if (variable == "hopping_x") {
				parameters.hopping_x = std::stod(value);
			} else if (variable == "hopping_lz") {
				parameters.hopping_lz = std::stod(value);
			} else if (variable == "hopping_ly") {
				parameters.hopping_ly = std::stod(value);
			} else if (variable == "hopping_lx") {
				parameters.hopping_lx = std::stod(value);
			} else if (variable == "hopping_rz") {
				parameters.hopping_rz = std::stod(value);
			} else if (variable == "hopping_ry") {
				parameters.hopping_ry = std::stod(value);
			} else if (variable == "hopping_rx") {
				parameters.hopping_rx = std::stod(value);
			} else if (variable == "hopping_lc") {
				parameters.hopping_lc = std::stod(value);
			} else if (variable == "hopping_rc") {
				parameters.hopping_rc = std::stod(value);
			} else if (variable == "hopping_ins_l_cor") {
				parameters.hopping_ins_l_cor = std::stod(value);
			} else if (variable == "hopping_ins_r_cor") {
				parameters.hopping_ins_r_cor = std::stod(value);
			} else if (variable == "num_cor") {
				parameters.num_cor = std::stoi(value);
			} else if (variable == "num_ins_left") {
				parameters.num_ins_left = std::stod(value);
			} else if (variable == "num_ins_right") {
				parameters.num_ins_right = std::stod(value);
			} else if (variable == "ins_metal_ins") {
				std::istringstream(value) >> parameters.ins_metal_ins;
			} else if (variable == "num_ky_points") {
				parameters.num_ky_points = std::stoi(value);
			} else if (variable == "num_kx_points") {
				parameters.num_kx_points = std::stoi(value);
			} else if (variable == "chemical_potential") {
				parameters.chemical_potential = std::stod(value);
			} else if (variable == "temperature") {
				parameters.temperature = std::stod(value);
			} else if (variable == "e_upper_bound") {
				parameters.e_upper_bound = std::stod(value);
			} else if (variable == "e_lower_bound") {
				parameters.e_lower_bound = std::stod(value);
			} else if (variable == "hubbard_interaction") {
				parameters.hubbard_interaction = std::stod(value);
			} else if (variable == "voltage_step") {
				parameters.voltage_step = std::stoi(value);
			} else if (variable == "self_consistent_steps") {
				parameters.self_consistent_steps = std::stod(value);
			} else if (variable == "read_in_self_energy") {
				std::istringstream(value) >> parameters.read_in_self_energy;
			} else if (variable == "NIV_points") {
				parameters.NIV_points = std::stoi(value);
			} else if (variable == "NIV_start") {
                parameters.NIV_start = std::stoi(value);
            } else if (variable == "delta_v") {
				parameters.delta_v = std::stod(value);
			} else if (variable == "delta_leads") {
				parameters.delta_leads = std::stod(value);
			} else if (variable == "delta_gf") {
				parameters.delta_gf = std::stod(value);
			} else if (variable == "leads_3d") {
				std::istringstream(value) >> parameters.leads_3d;
			} else if (variable == "spin_up_occup") {
				parameters.spin_up_occup = std::stod(value);
			} else if (variable == "spin_down_occup") {
				parameters.spin_down_occup = std::stod(value);
			} else if (variable == "convergence") {
				parameters.convergence = std::stod(value);
			} else if (variable == "gamma") {
				parameters.gamma = std::stod(value);
			} else if (variable == "wbl_approx") {
				std::istringstream(value) >> parameters.wbl_approx;
			} //else if (variable == "kk_relation") {
				//std::istringstream(value) >> parameters.kk_relation;
			//} 
			  else if (variable == "steps") {
				parameters.steps = std::stoi(value);
			} else if (variable == "print_gf") {
				std::istringstream(value) >> parameters.print_gf;
			} else if (variable == "interaction_order") {
				parameters.interaction_order = std::stoi(value);
			} else if (variable == "spin_polarised") {
				std::istringstream(value) >> parameters.spin_polarised;
			} else if (variable == "noneq_test") {
				std::istringstream(value) >> parameters.noneq_test;
			} else if (variable == "impurity_solver") {
                parameters.impurity_solver = std::stoi(value);
            }
	}
	input_file.close();
	

	parameters.path_of_self_energy_up = "textfiles/local_se_up_1_k_points_81_energy.txt";
	parameters.path_of_self_energy_down = "textfiles/local_se_down_1_k_points_81_energy.txt";

	parameters.voltage_l.resize(parameters.NIV_points);
	parameters.voltage_r.resize(parameters.NIV_points);
	for (int i = 0; i < parameters.NIV_points; i++) {
		parameters.voltage_l.at(i) = parameters.delta_v * (double)(i);
		parameters.voltage_r.at(i) = -parameters.delta_v * (double)(i);
	}

	if (parameters.hubbard_interaction == 0.0) {
		parameters.interaction_order =
		    0;  // this is the order the green function will be calculated too in terms of interaction strength. this can be equal to 0 , 1 or 2//
	} 

	if (parameters.ins_metal_ins == true) {
		parameters.chain_length =
		    parameters.num_ins_left + parameters.num_ins_right + parameters.num_cor;
	} else {
		parameters.chain_length = parameters.num_ins_left + 2 * parameters.num_cor;
	}

	//the atom is correlated if atom typ equals to 1. This how we know to apply sigma two to certain atoms.
	if (parameters.ins_metal_ins == true) {
		//we do this 4 times as there are two horizontal and two veritcal layers within the unit cell.
		//atoms 0 to chain_length -1 are the top left layer.
		//the top right atoms  are chain_length to 2 * chain_length -1.
		//the bottom left are 2 *atoms number 2 * chain_length to 3 * chain_length - 1.
		//the bottom right are atoms 3 * * chain_length to 4 * chain_length - 1.
		for (int s = 0; s < 4; s++) {
			for (int i = 0; i < parameters.num_ins_left; i++) {
				parameters.atom_type.push_back(0);
			}
			for (int i = 0; i < parameters.num_cor; i++) {
				parameters.atom_type.push_back(1);
			}
			for (int i = 0; i < parameters.num_ins_right; i++) {
				parameters.atom_type.push_back(0);
			}
		}
	} else {
		for (int s = 0; s < 4; s++) {
			for (int i = 0; i < parameters.num_cor; i++) {
				parameters.atom_type.push_back(1);
			}
			for (int i = 0; i < parameters.num_ins_left; i++) {
				parameters.atom_type.push_back(0);
			}
			for (int i = 0; i < parameters.num_cor; i++) {
				parameters.atom_type.push_back(1);
			}
		}
	}

	//for (int i = 0; i < 4 * parameters.chain_length; i++) {
	//	std::cout << parameters.atom_type.at(i) << std::endl;
	//}

	parameters.energy.resize(parameters.steps);

	parameters.j1 = -1;
	parameters.j1 = sqrt(parameters.j1);

	double delta_energy =
	    (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;

	for (int i = 0; i < parameters.steps; i++) {
		parameters.energy.at(i) = parameters.e_lower_bound + delta_energy * (double)i;
	}
	return parameters;

	if (delta_energy < parameters.delta_v) {
		std::cout << "Delta voltage is less than delta energy. This gives unphysical step function "
		             "results. Make delta_energy < parameters.delta_v"
		          << std::endl;
	}
}

double fermi_function(double energy, const Parameters& parameters)
{
	if (parameters.temperature == 0) {
		if (energy < parameters.chemical_potential) {
			return 1.0;
		} else {
			return 0.0;
		}
	} else {
	}
	return 1.0 / (1.0 + exp((energy - parameters.chemical_potential) / parameters.temperature));
}
//Parameters params = Parameters::from_file();
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
	std::cout << "parameters.print_gf = " << parameters.print_gf << std::endl;
	std::cout << "parameters.spin_polarised = " << parameters.spin_polarised << std::endl;
	std::cout << "parameters.noneq_test = " << parameters.noneq_test << std::endl;
}
