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
     const double &z_prefactor, AIM &aim_up) {

    std::vector<double> boson_gf_lesser(parameters.steps), boson_gf_greater(parameters.steps);
    std::vector<double> fermion_gf_lesser(parameters.steps), fermion_gf_greater(parameters.steps);

    for (int r = 0; r < parameters.steps_myid; r++) {
        aim_up.impurity_gf_mb_lesser.at(r) = 0;
        aim_up.impurity_gf_mb_greater.at(r) = 0;
    }

	distribute_to_procs(parameters, fermion_gf_lesser, fermion.lesser_gf);
	distribute_to_procs(parameters, fermion_gf_greater, fermion.greater_gf);
	distribute_to_procs(parameters, boson_gf_lesser,  boson.lesser_gf);
	distribute_to_procs(parameters, boson_gf_greater,  boson.greater_gf);

	for (int r = 0; r < parameters.steps_myid; r++) {
		for (int i = 0; i < parameters.steps; i++) {
            int y = r + parameters.start.at(parameters.myid);
			if (((i + y) >= (parameters.steps / 2)) && ((i + y) < 3 * (parameters.steps / 2))) {
				//std::cout << r << " " << i << "  " << parameters.energy.at(r) + parameters.energy.at(i)  << "  " << parameters.energy.at(r + i - (parameters.steps / 2)) << "\n ";
				aim_up.impurity_gf_mb_lesser.at(r) -= boson_gf_greater.at(i) * fermion_gf_lesser.at(i + y - (parameters.steps / 2));
				aim_up.impurity_gf_mb_greater.at(r) -= boson_gf_lesser.at(i) * fermion_gf_greater.at(i + y - (parameters.steps / 2));				
			}		
		}
		//std::cout << parameters.steps / 2 << std::endl;
		aim_up.impurity_gf_mb_lesser.at(r) = aim_up.impurity_gf_mb_lesser.at(r) * z_prefactor / (2.0 * M_PI) * parameters.delta_energy;
		aim_up.impurity_gf_mb_greater.at(r) = aim_up.impurity_gf_mb_greater.at(r) * z_prefactor / (2.0 * M_PI) * parameters.delta_energy;		
	}

    //write_to_file(parameters, aim_up.impurity_gf_mb_lesser, "aim_up.impurity_gf_mb_lesser.dat", 0);
    //write_to_file(parameters, aim_up.impurity_gf_mb_greater, "aim_up.impurity_gf_mb_greater.dat", 0);   
}

void get_difference_self_energy(const Parameters &parameters, std::vector<dcomp> &self_energy_mb_up,
	 std::vector<dcomp> &old_self_energy_mb_up, double &difference, int &index){
	double difference_proc = - std::numeric_limits<double>::infinity();
	double old_difference = 0;
	double real_difference = 0, imag_difference = 0;
	for (int r = 0; r < parameters.steps_myid; r++) {
		real_difference = absolute_value(self_energy_mb_up.at(r).real() - old_self_energy_mb_up.at(r).real());
		imag_difference = absolute_value(self_energy_mb_up.at(r).imag() - old_self_energy_mb_up.at(r).imag());
		difference_proc = std::max(difference_proc, std::max(real_difference, imag_difference));
		old_self_energy_mb_up.at(r) = self_energy_mb_up.at(r);
		if (difference_proc > old_difference) {
			index = r;
		}
		old_difference = difference_proc;
	}
	//std::cout << "I am rank " << parameters.myid << ". The difference for me is " << difference_proc << std::endl;
	//MPI_Allreduce would do the same thing.
	MPI_Reduce(&difference_proc, &difference, 1, MPI_DOUBLE, MPI_MAX , 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&difference, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void intialise_gf(const Parameters &parameters, Pseudo_GF &boson, Pseudo_GF &fermion_up, Pseudo_GF &fermion_down) {
    //this is the initial guess for the gf
    int a = parameters.steps / 3;
    for (int r = 0; r < parameters.steps_myid; r++) {
        int y = r + parameters.start.at(parameters.myid);
        if ((y > a) && (y < 2 * a)) {
            double radius = 0.5 * (parameters.energy.at(2 * a) - parameters.energy.at(a));

            //std::cout << radius << "  " << (parameters.energy.at(y) - 3 * radius) << "\n";
            double x = - sqrt(radius * radius -
                 (parameters.energy.at(y)) * (parameters.energy.at(y)));
            boson.lesser_gf.at(r) = x;
            boson.greater_gf.at(r) =  x;
            fermion_up.greater_gf.at(r) = x;
            fermion_down.greater_gf.at(r) = x;
        }
    }
}


void get_greater_lesser_se_fermion(const Parameters &parameters, const Pseudo_GF &boson, Pseudo_GF &fermion, int voltage_step, AIM &aim_up) {
    //this code function will be for each spin and then we should call it twice for the spin polarised case. 
    fermion.lesser_se.clear();
    fermion.greater_se.clear();

    fermion.lesser_se.resize(parameters.steps_myid, 0);
    fermion.greater_se.resize(parameters.steps_myid, 0);

    std::vector<double> hybrisation_lesser(parameters.steps), hybrisation_greater(parameters.steps);
    std::vector<double> boson_gf_lesser(parameters.steps), boson_gf_greater(parameters.steps);

  
	distribute_to_procs(parameters, hybrisation_lesser,  aim_up.hybridisation_lesser);
	distribute_to_procs(parameters, hybrisation_greater,  aim_up.hybridisation_greater);
	distribute_to_procs(parameters, boson_gf_lesser,  boson.lesser_gf);
	distribute_to_procs(parameters, boson_gf_greater,  boson.greater_gf);

    //std::cout <<boson_gf_greater << "  " <<  hybrisation_greater.size() << std::endl;
    //std::cout << parameters.start.at(parameters.myid) << std::endl;

	for (int r = 0; r < parameters.steps_myid; r++) {
		for (int i = 0; i < parameters.steps; i++) {

            int y = r + parameters.start.at(parameters.myid);
                        
			if (((i + y) >= (parameters.steps / 2)) && ((i + y) < 3 * (parameters.steps / 2))) {
                //if (i + r) < (parameters.steps / 2), that means hybridisation is  approximately zero. 
                //((i + r) < 3 * (parameters.steps / 2), that means the boson gf is approximately  zero.
                // this method of integration has an error proportionaly to parameters.delta_energy as 
                //parameters.e_lower_bound + parameters.steps * paramters.delta_energy != - parameters.e_lower_bound. I am not clever enough to fix this.
                fermion.lesser_se.at(r) += - hybrisation_lesser.at(parameters.steps - i - 1) * boson_gf_lesser.at(i + y - parameters.steps / 2); 
                fermion.greater_se.at(r) += - hybrisation_greater.at(parameters.steps - i - 1) * boson_gf_greater.at(i + y - parameters.steps / 2);
			}		
		}
		//std::cout << parameters.steps / 2 << std::endl;
		fermion.greater_se.at(r) = fermion.greater_se.at(r) * parameters.delta_energy / (2.0 * M_PI);
		fermion.lesser_se.at(r) = fermion.lesser_se.at(r) * parameters.delta_energy / (2.0 * M_PI);
	}
}


void get_retarded_gf(const Parameters &parameters, Pseudo_GF &green_function) {
    
    green_function.retarded_gf.clear();
    green_function.retarded_gf.resize(parameters.steps, 0);  
    std::vector<double> imaginary_gf(parameters.steps), imaginary_gf_myid(parameters.steps_myid);


    for (int r = 0; r < parameters.steps_myid; r++) {
        imaginary_gf_myid.at(r) = 0.5 * green_function.greater_gf.at(r);
    }

	distribute_to_procs(parameters, imaginary_gf,  imaginary_gf_myid);

    for (int r = 0; r < parameters.steps_myid; r++) {
        int y = r + parameters.start.at(parameters.myid);
        double gf_real = kramer_kronig_relation(parameters, imaginary_gf, y);
        green_function.retarded_gf.at(r) = gf_real + imaginary_gf_myid.at(r) * parameters.j1;
    }
}


//void get_retarded_gf_fermion(const Parameters &parameters, Pseudo_GF &green_function) {   
//    for( int r = 0; r < parameters.steps_myid; r++) {
//        int y = parameters.start.at(parameters.myid) + r;
//        green_function.retarded_gf.at(r) = 1.0 / (parameters.energy.at(y) - parameters.onsite_cor - green_function.retarded_se.at(r));
//    }
//}

void get_greater_lesser_gf(const Parameters &parameters, Pseudo_GF &green_function) {
    for (int r = 0; r < parameters.steps_myid; r++) {
        green_function.greater_gf.at(r) = (green_function.retarded_gf.at(r) * green_function.greater_se.at(r) * std::conj(green_function.retarded_gf.at(r))).real();
        green_function.lesser_gf.at(r) = (green_function.retarded_gf.at(r) * green_function.lesser_se.at(r) * std::conj(green_function.retarded_gf.at(r))).real();
    }
}

void get_greater_lesser_se_boson(const Parameters &parameters, Pseudo_GF &boson, const Pseudo_GF &fermion_up,
    const Pseudo_GF &fermion_down, int voltage_step, AIM &aim_up, AIM &aim_down) {

    boson.lesser_se.clear();
    boson.greater_se.clear();

    boson.lesser_se.resize(parameters.steps_myid, 0);
    boson.greater_se.resize(parameters.steps_myid, 0);

    std::vector<double> hybrisation_lesser_up(parameters.steps), hybrisation_greater_up(parameters.steps);
    std::vector<double> fermion_gf_lesser_up(parameters.steps), fermion_gf_greater_up(parameters.steps);

    std::vector<double> hybrisation_lesser_down, hybrisation_greater_down;
    std::vector<double> fermion_gf_lesser_down, fermion_gf_greater_down;

	distribute_to_procs(parameters, hybrisation_lesser_up,  aim_up.hybridisation_lesser);
	distribute_to_procs(parameters, hybrisation_greater_up,  aim_up.hybridisation_greater);
	distribute_to_procs(parameters, fermion_gf_lesser_up,  fermion_up.lesser_gf);
	distribute_to_procs(parameters, fermion_gf_greater_up,  fermion_up.greater_gf);

    if (parameters.spin_polarised == 1) {

        hybrisation_lesser_down.resize(parameters.steps);
        hybrisation_greater_down.resize(parameters.steps);
        fermion_gf_lesser_down.resize(parameters.steps);
        fermion_gf_greater_down.resize(parameters.steps);

	    distribute_to_procs(parameters, hybrisation_lesser_down,  aim_up.hybridisation_lesser);
	    distribute_to_procs(parameters, hybrisation_greater_down,  aim_up.hybridisation_greater);
	    distribute_to_procs(parameters, fermion_gf_lesser_down,  fermion_down.lesser_gf);
	    distribute_to_procs(parameters, fermion_gf_greater_down,  fermion_down.lesser_gf);

        for (int r = 0; r < parameters.steps_myid; r++) {
		    for (int i = 0; i < parameters.steps; i++) {
                int y = r + parameters.start.at(parameters.myid);
		    	if (((i + y) >= (parameters.steps / 2)) && ((i + y) < 3 * (parameters.steps / 2))) {
                    //if (i + r) < (parameters.steps / 2), that means hybridisation is  approximately zero. 
                    //((i + r) < 3 * (parameters.steps / 2), that means the boson gf is approximately  zero.
                    boson.greater_se.at(r) += hybrisation_lesser_up.at(i) * fermion_gf_greater_up.at(i + y - (parameters.steps / 2)) 
                        + hybrisation_lesser_down.at(i) * fermion_gf_greater_down.at(i + y - (parameters.steps / 2));

                    boson.lesser_se.at(r) += hybrisation_greater_up.at(i) * fermion_gf_lesser_up.at(i + y - (parameters.steps / 2)) 
                        + hybrisation_greater_down.at(i) * fermion_gf_lesser_down.at(i + y - (parameters.steps / 2));
		    	}		
		    }
            //std::cout << parameters.steps / 2 << std::endl;
		    boson.greater_se.at(r) = boson.greater_se.at(r) * parameters.delta_energy / (2.0 * M_PI);
		    boson.lesser_se.at(r) = boson.lesser_se.at(r) * parameters.delta_energy / (2.0 * M_PI);
        }
    } else {
        for (int r = 0; r < parameters.steps_myid; r++) {
		    for (int i = 0; i < parameters.steps; i++) {
                int y = r + parameters.start.at(parameters.myid);
		    	if (((i + y) >= (parameters.steps / 2)) && ((i + y) < 3 * (parameters.steps / 2))) {
                    //if (i + r) < (parameters.steps / 2), that means hybridisation is  approximately zero. 
                    //((i + r) < 3 * (parameters.steps / 2), that means the boson gf is approximately  zero.
                    boson.greater_se.at(r) += hybrisation_lesser_up.at(i) * fermion_gf_greater_up.at(i + y - (parameters.steps / 2));

                    boson.lesser_se.at(r) += hybrisation_greater_up.at(i) * fermion_gf_lesser_up.at(i + y - (parameters.steps / 2));
		    	}		
		    }
            //std::cout << parameters.steps / 2 << std::endl;
		    boson.greater_se.at(r) = 2.0 * boson.greater_se.at(r) * parameters.delta_energy / (2.0 * M_PI);
		    boson.lesser_se.at(r) = 2.0 * boson.lesser_se.at(r) * parameters.delta_energy / (2.0 * M_PI);
        }
    }
}


void test_retarded_gf(const Parameters &parameters, Pseudo_GF &boson, Pseudo_GF &fermion_up, Pseudo_GF &fermion_down) {

    double test_boson_myid = 0.0, test_fermion_up_myid = 0.0, test_fermion_down_myid = 0.0;
    double test_boson = 0.0, test_fermion_up = 0.0, test_fermion_down = 0.0;

    if (parameters.spin_polarised == 1) {
        for (int r = 0; r < parameters.steps_myid; r++) {
            test_boson_myid += boson.retarded_gf.at(r).imag();
            test_fermion_up_myid += fermion_up.retarded_gf.at(r).imag();
            test_fermion_down_myid += fermion_down.retarded_gf.at(r).imag();
        }
    } else {
        for (int r = 0; r < parameters.steps_myid; r++) {
            test_boson_myid += boson.retarded_gf.at(r).imag();
            test_fermion_up_myid += fermion_up.retarded_gf.at(r).imag();
        }       
    }

    MPI_Reduce(&test_boson_myid, &test_boson, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&test_fermion_up_myid, &test_fermion_up, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&test_fermion_down_myid, &test_fermion_down, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (parameters.myid == 0) {
        std::cout << "The imaginary part of the boson retarded gf integrates to " <<  test_boson * -1.0 * parameters.delta_energy / (M_PI) << "\n";
        std::cout << "The imaginary part of the fermion up retarded gf integrates to " <<  test_fermion_up * -1.0 * parameters.delta_energy / (M_PI) << "\n";
        std::cout << "The imaginary part of the fermion down retarded gf integrates to " <<  test_fermion_up * -1.0 * parameters.delta_energy / (M_PI) << "\n";
    }
}


void get_retarded_gf_boson(const Parameters &parameters, Pseudo_GF &green_function) {   
    for( int r = 0; r < parameters.steps_myid; r++) {
        int y = parameters.start.at(parameters.myid) + r;
        green_function.retarded_gf.at(r) = 1.0 / (parameters.energy.at(y) - green_function.retarded_se.at(r));
    }
}

void get_greater_lesser_gf(const Parameters &parameters, Pseudo_GF &green_function, int count) {
    if (count < 8) {
        for (int r = 0; r < parameters.steps_myid; r++) {
            green_function.greater_gf.at(r) = 0.5 * green_function.greater_gf.at(r) + 0.5 * (green_function.retarded_gf.at(r) * green_function.greater_se.at(r) * std::conj(green_function.retarded_gf.at(r))).real();
            green_function.lesser_gf.at(r) = 0.5 * green_function.lesser_gf.at(r) + 0.5 * (green_function.retarded_gf.at(r) * green_function.lesser_se.at(r) * std::conj(green_function.retarded_gf.at(r))).real();
        }
    }  else {
        for (int r = 0; r < parameters.steps; r++) {
            green_function.greater_gf.at(r) = 0.9 * green_function.greater_gf.at(r) + 0.1 * (green_function.retarded_gf.at(r) * green_function.greater_se.at(r) * std::conj(green_function.retarded_gf.at(r))).real();
            green_function.lesser_gf.at(r) = 0.9 * green_function.lesser_gf.at(r) + 0.1 * (green_function.retarded_gf.at(r) * green_function.lesser_se.at(r) * std::conj(green_function.retarded_gf.at(r))).real();
        }
    }
}

double get_z_prefactor(const Parameters &parameters, Pseudo_GF &boson, Pseudo_GF &fermion_up, Pseudo_GF &fermion_down) {
    double z_prefactor = 0, z_prefactor_my_id = 0.0;

    if (parameters.spin_polarised == 1) {
        for (int r = 0; r < parameters.steps_myid; r++) {
            z_prefactor_my_id += - boson.lesser_gf.at(r) + fermion_up.lesser_gf.at(r) + fermion_down.lesser_gf.at(r);
        }
    } else {
        for (int r = 0; r < parameters.steps_myid; r++) {
            z_prefactor_my_id += - boson.lesser_gf.at(r) + 2.0 * fermion_up.lesser_gf.at(r);
        }
    }



    MPI_Reduce(&z_prefactor_my_id, &z_prefactor, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&z_prefactor, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return  parameters.delta_energy * z_prefactor / (2.0 * M_PI);
}

void get_retarded_impurity_se(const Parameters &parameters, double z_prefactor, AIM &aim_up) {
    std::vector<double> imaginary_gf(parameters.steps), imaginary_gf_myid(parameters.steps_myid);
    for (int r = 0; r < parameters.steps_myid; r++) {
        imaginary_gf_myid.at(r) = - 0.5 * (aim_up.impurity_gf_mb_lesser.at(r) - aim_up.impurity_gf_mb_greater.at(r)) / z_prefactor;
    }

    distribute_to_procs(parameters, imaginary_gf, imaginary_gf_myid);

    for (int r = 0; r < parameters.steps_myid; r++) {
        int y = r + parameters.start.at(parameters.myid);
        double re_gf = kramer_kronig_relation(parameters, imaginary_gf, y);
        aim_up.impurity_gf_mb_retarded.at(r) = re_gf + parameters.j1 * imaginary_gf_myid.at(r);
        aim_up.self_energy_mb_retarded.at(r) = (1.0 / aim_up.dynamical_field_retarded.at(r)) - 1.0 /(aim_up.impurity_gf_mb_retarded.at(r));
    }

    //write_to_file(parameters, aim_up.impurity_gf_mb_retarded, "aim_up.impurity_gf_mb_retarded", 0);
    //write_to_file(parameters, aim_up.self_energy_mb_retarded, "aim_up.self_energy_mb_retarded", 0);
}

void integrate_dos(const Parameters &parameters, Pseudo_GF &fermion_up, double &z_prefactor, AIM &aim_up) {
    double dos = 0, dos_myid = 0, fermion_occup = 0, fermion_occup_myid = 0;
    
    for (int r = 0; r < parameters.steps_myid; r++) {
        dos_myid += (aim_up.impurity_gf_mb_lesser.at(r) - aim_up.impurity_gf_mb_greater.at(r));
        fermion_occup_myid +=  fermion_up.lesser_gf.at(r);
    }


    MPI_Reduce(&dos_myid, &dos, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&fermion_occup_myid, &fermion_occup, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);



    if (parameters.myid == 0) {
        std::cout << "The sum of the dos and fermion occupation is " << dos * parameters.delta_energy / (2.0 * M_PI) +
             fermion_occup * parameters.delta_energy / (2.0 * M_PI) * z_prefactor << "\n";
    }
}

void get_lesser_greater_impurity_se(const Parameters &parameters, AIM &aim_up) {
    for (int r = 0; r < parameters.steps_myid; r++) {
        aim_up.self_energy_mb_lesser.at(r) = ((1.0 / aim_up.impurity_gf_mb_retarded.at(r)) * aim_up.impurity_gf_mb_lesser.at(r) 
            * (1.0 / std::conj(aim_up.impurity_gf_mb_retarded.at(r)))).real() - aim_up.hybridisation_lesser.at(r);
        aim_up.self_energy_mb_greater.at(r) = ((1.0 / aim_up.impurity_gf_mb_retarded.at(r)) * aim_up.impurity_gf_mb_greater.at(r) 
            * (1.0 / std::conj(aim_up.impurity_gf_mb_retarded.at(r)))).real() - aim_up.hybridisation_greater.at(r);
    }
}

void solve_pseudo_particle_gf(const Parameters &parameters, Pseudo_GF &boson, Pseudo_GF &fermion_up, Pseudo_GF &fermion_down, int voltage_step,
     double &z_prefactor, AIM &aim_up, AIM &aim_down) {

    std::vector<dcomp> old_fermion_gf(parameters.steps_myid, 0);
    int count = 0;
    double difference = std::numeric_limits<double>::infinity();
    std::vector<double> coupling;

    intialise_gf(parameters, boson, fermion_up, fermion_down);
    while (difference > parameters.convergence && count < parameters.self_consistent_steps_nca) {
        if (parameters.spin_polarised == 1) {
            
            get_greater_lesser_se_fermion(parameters, boson, fermion_up, voltage_step, aim_up);
            get_greater_lesser_se_fermion(parameters, boson, fermion_down, voltage_step, aim_down);

            get_retarded_gf(parameters, fermion_up);
            get_retarded_gf(parameters, fermion_down);

            get_greater_lesser_gf(parameters, fermion_up);
            get_greater_lesser_gf(parameters, fermion_down);

            get_greater_lesser_se_boson(parameters, boson, fermion_up, fermion_down, voltage_step, aim_up, aim_down);
            
            get_retarded_gf(parameters, boson);
            
            get_greater_lesser_gf(parameters, boson, count);
        } else {

            get_greater_lesser_se_fermion(parameters, boson, fermion_up, voltage_step, aim_up);

            get_retarded_gf(parameters, fermion_up);

            get_greater_lesser_gf(parameters, fermion_up);

            get_greater_lesser_se_boson(parameters, boson, fermion_up, fermion_down, voltage_step, aim_up, aim_down);
            
            get_retarded_gf(parameters, boson);
            
            get_greater_lesser_gf(parameters, boson, count);
        }
        
        int index;
        get_difference_self_energy(parameters, fermion_up.retarded_gf, old_fermion_gf, difference, index);
        
        count++;
        if (parameters.myid == 0) {
            std::cout << "The count is " << count << ". The difference for the nca loop is " << difference << std::endl;
        }
        
    }

    //test_retarded_gf(parameters, boson, fermion_up, fermion_down);



    z_prefactor =  1.0 / get_z_prefactor(parameters, boson, fermion_up, fermion_down);
}




void impurity_solver_nca(const Parameters &parameters, const int voltage_step, AIM &aim_up, AIM &aim_down)
{
	Pseudo_GF boson(parameters);
	Pseudo_GF fermion_up(parameters);
	Pseudo_GF fermion_down(parameters);	


	double z_prefactor = 0;
	solve_pseudo_particle_gf(parameters, boson, fermion_up, fermion_down, voltage_step, z_prefactor, aim_up, aim_down);


	//if (parameters.myid) {
    //    //std::cout << "The ratio of Z_0 / Z_1 is " << z_prefactor << std::endl;
    //}
    if (parameters.spin_polarised == 1) {
        get_lesser_greater_gf(parameters, boson, fermion_up, z_prefactor, aim_up);
	    get_lesser_greater_gf(parameters, boson, fermion_down, z_prefactor, aim_down);
        get_retarded_impurity_se(parameters, z_prefactor, aim_up);
        get_retarded_impurity_se(parameters, z_prefactor, aim_down);
    } else {
        get_lesser_greater_gf(parameters, boson, fermion_up, z_prefactor, aim_up);
        get_retarded_impurity_se(parameters, z_prefactor, aim_up);
        integrate_dos(parameters, fermion_up, z_prefactor, aim_up);
    }

	boson.print_green_function(parameters, voltage_step, "boson");
	fermion_up.print_green_function(parameters, voltage_step, "fermion_up");
	//fermion_down.print_green_function(parameters, m, "fermion_down");
}


