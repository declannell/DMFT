#include "parameters.h"
#include "utilis.h"
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
	
	parameters.convergence_leads = 0.0001;
	parameters.self_consistent_steps_leads = 20;
	parameters.half_metal = 0;
	parameters.meir_wingreen_current = 1;
	parameters.bond_current = 0;
	std::vector<int> temp_interace;
	bool insideInterfacesBlock = false;
	while (getline(input_file, line)) {

		
		if (line == "interfaces_start") {
            insideInterfacesBlock = true;
        } else if (line == "interfaces_end") {
            insideInterfacesBlock = false;
        } else if (insideInterfacesBlock == true) {
            int interface_num = std::stoi(line);
            temp_interace.push_back(interface_num);
        } else {
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
			}else if (variable == "self_consistent_steps_nca") {
				parameters.self_consistent_steps_nca = std::stod(value);
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
			} else if (variable == "impurity_solver") {
        	    parameters.impurity_solver = std::stoi(value);
        	} else if (variable == "magnetic_field") {
				parameters.magnetic_field = std::stod(value);
			} else if (variable == "half_metal") {
				parameters.half_metal = std::stoi(value);
			} else if (variable == "convergence_leads") {
				parameters.convergence_leads = std::stod(value);
			} else if (variable == "self_consistent_steps_leads") {
				parameters.self_consistent_steps_leads = std::stod(value);
			} else if (variable == "bond_current") {
				parameters.bond_current = std::stoi(value);
			} else if (variable == "meir_wingreen_current") {
				parameters.meir_wingreen_current = std::stoi(value);
			} 
		}
	}
	input_file.close();
	
	parameters.interface.resize(temp_interace.size(), 0);
	for (long unsigned int i = 0; i < temp_interace.size(); i++) {
		parameters.interface.at(i) = temp_interace.at(i);
	}

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


	int atom_identity;
	if (parameters.half_metal == 1) {//the insulating layers are replaced by half metals.
		atom_identity = 2;
	} else if (parameters.half_metal == 0) {//the insulating remain insulting.
		atom_identity = 0;
	}

	//the atom is correlated if atom typ equals to 1. This how we know to apply sigma two to certain atoms.
	if (parameters.ins_metal_ins == true) {
		//we do this 4 times as there are two horizontal and two veritcal layers within the unit cell.
		//atoms 0 to chain_length -1 are the top left layer.
		//the top right atoms  are chain_length to 2 * chain_length -1.
		//the bottom left are 2 *atoms number 2 * chain_length to 3 * chain_length - 1.
		//the bottom right are atoms 3 * * chain_length to 4 * chain_length - 1.
		for (int s = 0; s < 4; s++) {//this is an array which encodes information about the i'th orbtial. Whether it is metallic or insulating, half metallic etc.
			for (int i = 0; i < parameters.num_ins_left; i++) {
				parameters.atom_type.push_back(atom_identity);
			}
			for (int i = 0; i < parameters.num_cor; i++) {
				parameters.atom_type.push_back(1);
			}
			for (int i = 0; i < parameters.num_ins_right; i++) {
				parameters.atom_type.push_back(atom_identity);
			}
		}
	} else {
		for (int s = 0; s < 4; s++) {
			for (int i = 0; i < parameters.num_cor; i++) {
				parameters.atom_type.push_back(1);
			}
			for (int i = 0; i < parameters.num_ins_left; i++) {
				parameters.atom_type.push_back(atom_identity);
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

	parameters.delta_energy =
	    (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;

	for (int i = 0; i < parameters.steps; i++) {
		parameters.energy.at(i) = parameters.e_lower_bound + parameters.delta_energy * (double)i;
	}

	MPI_Comm_size(MPI_COMM_WORLD, &parameters.size);
	MPI_Comm_rank(MPI_COMM_WORLD, &parameters.myid);
	parameters.comm = MPI_COMM_WORLD;

	if (parameters.myid == 0) {
		print_parameters(parameters);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (parameters.myid == 0) {
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;
	}
	
	parameters.steps_proc.resize(parameters.size, 0);
	parameters.end.resize(parameters.size, 0);
	parameters.start.resize(parameters.size, 0);
	
	decomp(parameters.steps, parameters.size, parameters.myid, &parameters.start.at(parameters.myid), &parameters.end.at(parameters.myid));
	parameters.steps_myid = parameters.end.at(parameters.myid) - parameters.start.at(parameters.myid) + 1;

    std::string message = "My myid is " + std::to_string(parameters.myid) + " in a world of size " +
                          std::to_string(parameters.size) + ". There are " + std::to_string(parameters.steps) +
                          " energy steps in my parameters class. The starting point and end point of my array are " +
                          std::to_string(parameters.start.at(parameters.myid)) + " and " +
                          std::to_string(parameters.end.at(parameters.myid)) + ". The number of points in my process are " +
                          std::to_string(parameters.steps_myid);

    for (int i = 0; i < parameters.size; i++) {
        MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes

        if (parameters.myid == i) {
            std::cout << std::setprecision(15) << message << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
    }
		
	parameters.steps_proc.at(parameters.myid) = parameters.steps_myid;

	//if (parameters.myid == 0) {
	//	if (parameters.interface > parameters.chain_length) {
	//		std::cout << "the chosen interface is not a valid choice as it is great than the number of layers \n";
	//		exit(1);
	//	}
	//	if ( parameters.interface < 1) {
	//		std::cout << "the chosen interface is not a valid choice as it is less than one \n";
	//		exit(1);
	//	}
	//	std::cout << std::endl;
	//	std::cout << std::endl;
	//	std::cout << std::endl;
	//}

	for (int a = 0; a < parameters.size; a++){
		MPI_Bcast(&parameters.start.at(a), 1, MPI_INT, a, MPI_COMM_WORLD);
		MPI_Bcast(&parameters.end.at(a), 1, MPI_INT, a, MPI_COMM_WORLD);
		MPI_Bcast(&parameters.steps_proc.at(a), 1, MPI_INT, a, MPI_COMM_WORLD);
	}


	if (parameters.magnetic_field != 0 || parameters.half_metal == 1) {
		parameters.spin_polarised = true;
	}

	if (parameters.delta_energy < parameters.delta_v) {
		if (parameters.myid == 0) std::cout << "Delta voltage is less than delta energy. This gives unphysical step function "
		             "results. Make delta_energy < parameters.delta_v" << std::endl;
	}

	return parameters;

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
	std::cout << "self_consistent_steps_nca = " << parameters.self_consistent_steps_nca << std::endl;
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
	std::cout << "parameters.magnetic_field = " << parameters.magnetic_field << std::endl;
	std::cout << "parameters.half_metal = " << parameters.half_metal << std::endl;
	std::cout << "parameters.convergence_leads = " << parameters.convergence_leads << std::endl;
	std::cout << "parameters.self_consistent_steps_leads = " << parameters.self_consistent_steps_leads << std::endl;
	std::cout << "paramters.meir_wingreen_current  = " << parameters.meir_wingreen_current << std::endl;
	std::cout << "parameters.bond_current = " << parameters.bond_current << std::endl;
	std::cout << "the parameters interface are ";
	for (long unsigned int i = 0; i < parameters.interface.size(); i++) {
		std::cout << i << ", ";
	}
	std::cout << "\n";
}
