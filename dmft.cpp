#include "parameters.h"
#include "leads_self_energy.h"
#include "dmft.h"
#include "interacting_gf.h"
#include <iostream>
#include <vector>
#include <x86_64-linux-gnu/mpich/mpi.h>
#include <eigen3/Eigen/Dense>



void get_spin_occupation(Parameters &parameters, std::vector<dcomp> &gf_lesser_up,
                        std::vector<dcomp> &gf_lesser_down, double *spin_up, double *spin_down){
    double delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound) / (double)parameters.steps;
    double result_up = 0.0, result_down = 0.0;
    for(int r = 0; r < parameters.steps; r++){
        result_up = (delta_energy) * gf_lesser_up.at(r).imag() + result_up;
        result_down = (delta_energy) * gf_lesser_down.at(r).imag() + result_down;
    }
    *spin_up = 1.0 / (2.0 * M_PI) * result_up;
    *spin_down = 1.0 / (2.0 * M_PI) * result_down;

    std::cout << *spin_up << std::endl;
}


void get_difference(Parameters &parameters, std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &old_green_function,
                double &difference){
    int n = parameters.chain_length * parameters.chain_length * parameters.steps;
    difference = -std::numeric_limits<double>::infinity();
    double real_difference, imag_difference;
    for (int r = 0; r < parameters.steps; r++) {
        for(int i = 0; i < parameters.chain_length; i++){
            for(int j = 0; j < parameters.chain_length; j++){
                real_difference = abs(gf_local_up.at(r)(i, j).real() - old_green_function.at(r)(i, j).real());
                imag_difference = abs(gf_local_up.at(r)(i, j).imag() - old_green_function.at(r)(i, j).imag());
                difference = std::max(difference, std::max(real_difference, imag_difference));
                old_green_function.at(r)(i, j) = gf_local_up.at(r)(i, j);
            }
        }
    }
}

void fluctuation_dissipation(Parameters &parameters, const std::vector<dcomp> &green_function, std::vector<dcomp> &lesser_green_function){
    for(int r = 0; r < parameters.steps; r++){
        lesser_green_function.at(r) = -fermi_function(parameters.energy.at(r).real(), parameters) * (
            green_function.at(r) - std::conj(green_function.at(r)));
    }
    std::ofstream myfile2;
    myfile2.open("/home/declan/green_function_code/quantum_transport/textfiles/gf_lesser_c++.txt");
    // myfile << parameters.steps << std::endl;
    for (int r = 0; r < parameters.steps; r++)
    {
        myfile2 << parameters.energy.at(r).real() << "," << lesser_green_function.at(r).real() << "," << lesser_green_function.at(r).imag() << "\n";
    }
    myfile2.close();    
}


dcomp integrate(Parameters &parameters, std::vector<dcomp> &gf_1, std::vector<dcomp> &gf_2,
            std::vector<dcomp> &gf_3, int r){
    double delta_energy = (parameters.e_upper_bound -
                    parameters.e_lower_bound) / (double)parameters.steps;
    dcomp result = 0;
    for(int i = 0; i < parameters.steps; i++) {
        for(int j = 0; j < parameters.steps; j++){
            if (((i + j - r) >= 0) && ((i + j - r) < parameters.steps)) {
                //this integrates like PHYSICAL REVIEW B 74, 155125 2006
                //I say the green function is zero outside e_lower_bound and e_upper_bound. This means I need the final green function in the integral to be within an energy of e_lower_bound
                //and e_upper_bound. The index of 0 corresponds to e_lower_bound. Hence we need i+J-r>0 but in order to be less an energy of e_upper_bound we need i+j-r<steps. 
                //These conditions enesure the enrgy of the gf3 greens function to be within (e_upper_bound, e_lower_bound)
                result = (delta_energy / (2.0 * parameters.pi)) * (delta_energy / (2.0 * parameters.pi)) * 
                    gf_1.at(i) * gf_2.at(j) * gf_3.at(i + j - r) + result;
            } 
        }
    }
    return result;
}

void self_energy_2nd_order(Parameters &parameters, std::vector<dcomp> &impurity_gf_up, std::vector<dcomp> &impurity_gf_down, 
            std::vector<dcomp> &impurity_gf_up_lesser, std::vector<dcomp> &impurity_gf_down_lesser, std::vector<dcomp> &impurity_self_energy){
    
    std::vector<dcomp> impurity_gf_down_advanced(parameters.steps);
    for(int r = 0; r < parameters.steps; r++){
        impurity_gf_down_advanced.at(r) = std::conj(impurity_gf_down.at(r));
    }
    for(int r = 0; r < parameters.steps; r++){
        impurity_self_energy.at(r) = parameters.hubbard_interaction * parameters.hubbard_interaction * 
            (integrate(parameters, impurity_gf_up, impurity_gf_down,impurity_gf_down_lesser, r));  // line 3

        impurity_self_energy.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction *
            (integrate(parameters, impurity_gf_up, impurity_gf_down_lesser, impurity_gf_down_lesser, r));  // line 2

        impurity_self_energy.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction * 
        (integrate(parameters, impurity_gf_up_lesser, impurity_gf_down, impurity_gf_down_lesser, r));  // line 1

        impurity_self_energy.at(r) += parameters.hubbard_interaction * parameters.hubbard_interaction * 
        (integrate(parameters, impurity_gf_up_lesser, impurity_gf_down_lesser, impurity_gf_down_advanced, r));  //line 4
    }
}


void impurity_solver(Parameters &parameters, std::vector<dcomp>  &impurity_gf_up, std::vector<dcomp>  &impurity_gf_down,
        std::vector<dcomp>  &impurity_self_energy_up, std::vector<dcomp>  &impurity_self_energy_down){
    std::vector<dcomp> impurity_gf_up_lesser(parameters.steps, 0), impurity_gf_down_lesser(parameters.steps, 0);

    if (parameters.voltage_step == 0) {
        fluctuation_dissipation(parameters, impurity_gf_up, impurity_gf_up_lesser);
        fluctuation_dissipation(parameters, impurity_gf_down, impurity_gf_down_lesser);
    }

    double impurity_spin_up, impurity_spin_down;
    get_spin_occupation(parameters, impurity_gf_up_lesser, impurity_gf_down_lesser, &impurity_spin_up, &impurity_spin_down);
    std::cout << "The spin up occupancy is " <<impurity_spin_up << "\n";
    std::cout << "The spin down occupancy is " <<impurity_spin_down << "\n";
    
    if (parameters.interaction_order == 2){
        self_energy_2nd_order(parameters, impurity_gf_up, impurity_gf_down, impurity_gf_up_lesser,
            impurity_gf_down_lesser, impurity_self_energy_up);
        self_energy_2nd_order(parameters, impurity_gf_down, impurity_gf_up, impurity_gf_down_lesser,
            impurity_gf_up_lesser, impurity_self_energy_down);

        for(int r = 0; r < parameters.steps; r++){
            impurity_self_energy_up.at(r) += parameters.hubbard_interaction * impurity_spin_down;
            impurity_self_energy_down.at(r) += parameters.hubbard_interaction * impurity_spin_up;
        }
    }

    if (parameters.interaction_order == 1){
        for(int r = 0; r < parameters.steps; r++) {
            impurity_self_energy_up.at(r) = parameters.hubbard_interaction * impurity_spin_down;
            impurity_self_energy_down.at(r) = parameters.hubbard_interaction * impurity_spin_up;
        }
    }
}



void dmft(Parameters &parameters, int voltage_step, std::vector<double> const &kx, std::vector<double> const &ky, 
        std::vector<std::vector<dcomp>> &self_energy_mb_up, std::vector<std::vector<dcomp>> &self_energy_mb_down, 
        std::vector<Eigen::MatrixXcd> &gf_local_up, std::vector<Eigen::MatrixXcd> &gf_local_down, 
        std::vector<std::vector<EmbeddingSelfEnergy>> &leads){

    double difference = std::numeric_limits<double>::infinity();
    int count = 0;
    std::vector<Eigen::MatrixXcd> old_green_function(parameters.steps, Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
    while (difference > 0.0001 && count < parameters.self_consistent_steps){
        get_difference(parameters, gf_local_up, old_green_function, difference);
        std::cout << "The difference is " << difference <<". The count is " << count << std::endl;
        if (difference < 0.0001){
            break;
        }
        if (parameters.interaction_order != 0.0) {
            for(int i = 0; i < parameters.chain_length; i++) {
                std::vector<dcomp> diag_gf_local_up(parameters.steps), diag_gf_local_down(parameters.steps), 
                    impurity_self_energy_up(parameters.steps), impurity_self_energy_down(parameters.steps);

                for(int r = 0; r < parameters.steps; r++){
                    diag_gf_local_up.at(r) = gf_local_up.at(r)(i, i);
                    diag_gf_local_down.at(r) = gf_local_down.at(r)(i, i);
                }

                impurity_solver(parameters, diag_gf_local_up, diag_gf_local_down, impurity_self_energy_up, impurity_self_energy_down);

                for(int r = 0; r < parameters.steps; r++){
                    self_energy_mb_up.at(i).at(r) = impurity_self_energy_up.at(r);
                    self_energy_mb_down.at(i).at(r) = impurity_self_energy_down.at(r);
                }
            }
            get_local_gf(parameters, kx, ky, self_energy_mb_up, leads, gf_local_up, gf_local_down);
        } else {
            break;
        }
        count++;
    }
}


/*
def get_coupling_matrices(kx: int, ky: int):
    self_energy = leads_self_energy.EmbeddingSelfEnergy(
        kx, ky, parameters.voltage_step)
    coupling_left, coupling_right = [0 for i in range(0, parameters.steps)], [
        0 for i in range(0, parameters.steps)
    ]
    for r in range(0, parameters.steps):
        coupling_left[r] = 1j * (
            self_energy.self_energy_left[r] -
            parameters.conjugate(self_energy.self_energy_left[r]))
        coupling_right[r] = 1j * (
            self_energy.self_energy_right[r] -
            parameters.conjugate(self_energy.self_energy_right[r]))

    return coupling_left, coupling_right

def self_energy_from_textfile(voltage: int, kx: List[float], ky: List[float]):
    print("reading in the textfile")
    self_energy_mb_up = [[0 for i in range(0, parameters.chain_length)]
                         for z in range(0, parameters.steps)]
    self_energy_mb_down = [[0 for i in range(0, parameters.chain_length)]
                           for z in range(0, parameters.steps)]
    lines_complex = [[0, 0] for r in range(0, parameters.steps)]

    embedding_self_energy = [[
        leads_self_energy.EmbeddingSelfEnergy(kx[i], kx[j],
                                              parameters.voltage_step)
        for j in range(0, parameters.chain_length_y)
    ] for i in range(0, parameters.chain_length_x)]

    with open(parameters.path_of_self_energy_up, 'r',
              encoding='utf-8') as infile:
        lines = infile.read().rsplit()
        for r in range(0, parameters.steps):
            for i in range(0, parameters.chain_length):
                lines_complex[r] = lines[r].split(',')
                self_energy_mb_up[r][i] = float(
                    lines_complex[r][0]) + 1j * float(lines_complex[r][1])

    with open(parameters.path_of_self_energy_down, 'r',
              encoding='utf-8') as infile:
        lines = infile.read().rsplit()
        for r in range(0, parameters.steps):
            for i in range(0, parameters.chain_length):
                lines_complex[r] = lines[r].split(',')
                self_energy_mb_down[r][i] = float(
                    lines_complex[r][0]) + 1j * float(lines_complex[r][1])

    gf_local_up, gf_local_down = get_local_gf(kx, ky, self_energy_mb_up,
                                              self_energy_mb_down,
                                              embedding_self_energy)

    plot_and_write_files(gf_local_up, gf_local_down, self_energy_mb_up,
                         self_energy_mb_down)

    return gf_local_up, gf_local_down  # , spin_up_occup, spin_down_occup



def plot_and_write_files(gf_local_up: List[List[List[complex]]],
                         gf_local_down: List[List[List[complex]]],
                         self_energy_mb_up: List[List[complex]],
                         self_energy_mb_down: List[List[complex]]):

    parser = argparse.ArgumentParser()

    parser.add_argument("-tf", "--textfile", help="Textfiles", type=bool)

    args = parser.parse_args()
    if (args.textfile == True):
        f = open(
            '/home/declan/green_function_code/quantum_transport/textfiles/local_gf_%i_k_points_%i_energy.txt'
            % (parameters.chain_length_x, parameters.steps), 'w')
        for r in range(0, parameters.steps):
            f.write(str(gf_local_up[r][0][0].real))
            f.write(",")
            f.write(str(gf_local_up[r][0][0].imag))
            f.write("\n")
        f.close()

        if (parameters.hubbard_interaction != 0):
            f = open(
                '/home/declan/green_function_code/quantum_transport/textfiles/local_se_up_%i_k_points_%i_energy.txt'
                % (parameters.chain_length_x, parameters.steps), 'w')
            for r in range(0, parameters.steps):
                f.write(str(self_energy_mb_up[r][0].real))
                f.write(",")
                f.write(str(self_energy_mb_up[r][0].imag))
                f.write("\n")
            f.close()

        if (parameters.hubbard_interaction != 0):
            f = open(
                '/home/declan/green_function_code/quantum_transport/textfiles/local_se_down_%i_k_points_%i_energy.txt'
                % (parameters.chain_length_x, parameters.steps), 'w')
            for r in range(0, parameters.steps):
                f.write(str(self_energy_mb_down[r][0].real))
                f.write(",")
                f.write(str(self_energy_mb_down[r][0].imag))
                f.write("\n")
            f.close()

    for i in range(0, parameters.chain_length):
        plt.plot(parameters.energy, [e[i][i].real for e in gf_local_up],
                 color='red',
                 label='Real Green up')
        plt.plot(parameters.energy, [e[i][i].imag for e in gf_local_up],
                 color='blue',
                 label='Imaginary Green function')
        j = i + 1
        plt.title(
            'The local Green function site % i for %i k points and %i energy points'
            % (j, parameters.chain_length_x, parameters.steps))
        plt.legend(loc='upper left')
        plt.xlabel("energy")
        if (parameters.hubbard_interaction == 0):
            plt.ylabel("Noninteracting green Function")
        else:
            plt.ylabel("Interacting green Function")
        plt.show()

    if (parameters.hubbard_interaction != 0):
        for i in range(0, parameters.chain_length):
            fig = plt.figure()
            plt.plot(parameters.energy,
                     [e[i].imag for e in self_energy_mb_down],
                     color='blue',
                     label='imaginary self energy')
            # plt.plot(parameters.energy, [
            # e[i].real for e in self_energy_mb_down], color='red', label='real self energy')
            j = i + 1
            plt.title(
                'The local self energy site % i (%i k %i energy points)' %
                (j, parameters.chain_length_x, parameters.steps))
            plt.legend(loc='upper right')
            plt.xlabel("energy")
            plt.ylabel("Self Energy")
            plt.show()

        for i in range(0, parameters.chain_length):
            fig = plt.figure()
            plt.plot(parameters.energy, [e[i].imag for e in self_energy_mb_up],
                     color='blue',
                     label='imaginary self energy')
            plt.plot(parameters.energy, [e[i].real for e in self_energy_mb_up],
                     color='red',
                     label='real self energy')
            j = i + 1
            plt.title('Many-body self energy spin up site %i' % j)
            plt.legend(loc='upper right')
            plt.xlabel("energy")
            plt.ylabel("Self Energy")
            plt.show()

    #print("The spin up occupaton probability is ", spin_up_occup)
    #print("The spin down occupaton probability is ", spin_down_occup)
    # if(voltage == 0):#this compares the two methods in equilibrium
    #compare_g_lesser(gf_int_lesser_up , gf_int_up)


def first_order_self_energy(gf_local_up: List[complex],
                            gf_local_down: List[complex]):
    gf_up_lesser = fluctuation_dissipation(gf_local_up)
    gf_down_lesser = fluctuation_dissipation(gf_local_down)
    spin_up_occup, spin_down_occup = get_spin_occupation(
        gf_up_lesser, gf_down_lesser)
    return spin_up_occup, spin_down_occup


# this the analytic soltuion for the noninteracting green function when we have a single site in the scattering region
def analytic_local_gf_1site(gf_int_up: List[List[List[complex]]],
                            kx: List[float], ky: List[float]):
    # this assume the interaction between the scattering region and leads is nearest neighbour
    analytic_gf = [0 for i in range(parameters.steps)]
    for i in range(0, parameters.chain_length_x):
        for j in range(0, parameters.chain_length_y):
            self_energy = leads_self_energy.EmbeddingSelfEnergy(
                kx[i], ky[j], parameters.voltage_step)
            # self_energy.plot_self_energy()
            num_k_points = parameters.chain_length_x * parameters.chain_length_y
            for r in range(0, parameters.steps):
                x = parameters.energy[r].real - parameters.onsite - 2 * parameters.hopping_x * math.cos(kx[i]) \
                    - 2 * parameters.hopping_y * math.cos(ky[j]) - self_energy.self_energy_left[r].real - \
                    self_energy.self_energy_right[r].real

                y = self_energy.self_energy_left[r].imag + \
                    self_energy.self_energy_right[r].imag
                analytic_gf[r] += 1 / num_k_points * (x / (x * x + y * y) +
                                                      1j * y / (x * x + y * y))

    # plt.plot(parameters.energy, [
    # e[0][0].real for e in gf_int_up], color='red', label='real green function')
    plt.plot(parameters.energy, [-e[0][0].imag for e in gf_int_up],
             color='blue',
             label='imaginary green function')
    plt.plot(parameters.energy, [-e.imag for e in analytic_gf],
             color='blue',
             label='analytic imaginary green function')
    # plt.plot(parameters.energy, [e.real for e in analytic_gf],
    # color='red', label='analytic real green function')
    plt.title(" Analytical Green function and numerical GF")
    #plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Green Function")
    plt.show()

def analytic_gf_2site():#this the analytic soltuion for the noninteracting green function when we have 2 sites in the scattering region
    analytic_gf= [ 0 for i  in range( parameters.steps ) ]# this assume the interaction between the scattering region and leads is nearest neighbour
    energy = [ 0 for i in range( parameters.steps ) ]  
   
    self_energy = leads_self_energy.EmbeddingSelfEnergy(
        parameters.pi/2, parameters.pi/2, parameters.voltage_step)
    for r in range( 0 , parameters.steps ):  
        x= energy[r] - parameters.onsite - self_energy.self_energy_left[r].real
        y = self_energy.self_energy_left[r].imag
        a = x * x - y * y - parameters.hopping * parameters.hopping
        b = 2 * x * y
        analytic_gf[r] =  ( a * x + b * y ) / ( a * a + b * b ) + 1j * ( y * a - x * b ) / ( a * a + b * b )

   
    plt.plot(parameters.energy, [ e.imag for e in analytic_gf ], color='blue', label='imaginary green function' )
    plt.plot(parameters.energy, [e.real for e in analytic_gf] , color='red' , label='real green function')
    plt.title(" Analytical Green function")
    #plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Green Function")  
    plt.show()

def main():
    #h = hpy()
    kx = [0 for m in range(0, parameters.chain_length_x)]
    ky = [0 for m in range(0, parameters.chain_length_y)]
    for i in range(0, parameters.chain_length_y):
        if (parameters.chain_length_y != 1):
            ky[i] = 2 * parameters.pi * i / parameters.chain_length_y
        elif (parameters.chain_length_y == 1):
            ky[i] = parameters.pi / 2.0

    for i in range(0, parameters.chain_length_x):
        if (parameters.chain_length_x != 1):
            kx[i] = 2 * parameters.pi * i / parameters.chain_length_x
        elif (parameters.chain_length_x == 1):
            kx[i] = parameters.pi / 2.0
    print(ky)
    # voltage step of zero is equilibrium.
    print(
        "The voltage difference is ",
        parameters.voltage_l[parameters.voltage_step] -
        parameters.voltage_r[parameters.voltage_step])
    print("The number of sites in the z direction is ",
          parameters.chain_length)
    print("The number of sites in the x direction is ",
          parameters.chain_length_x)
    print("The number of sites in the y direction is ",
          parameters.chain_length_y)
    #print("The ky value is ", ky)
    #print("The kx value is ", kx)
    time_start = time.perf_counter()
    if (parameters.read_in_self_energy == True):
        green_function_up, green_function_down = self_energy_from_textfile(
            parameters.voltage_step, kx, ky)
    else:
        green_function_up, green_function_down = dmft(parameters.voltage_step,
                                                      kx, ky)
    if (parameters.chain_length == 1 and parameters.hubbard_interaction == 0):
        analytic_local_gf_1site(green_function_up, kx, ky)
    elif (parameters.chain_length == 2 and parameters.hubbard_interaction == 0):
        analytic_gf_2site()
    else:
        for i in range(0, parameters.chain_length):
            plt.plot(parameters.energy,
                     [-e[i][i].imag for e in green_function_up],
                     color='blue',
                     label='Imaginary Green function')
            j = i + 1
            plt.title(
                'The local Green function site % i for %i k points and %i energy points'
                % (j, parameters.chain_length_x, parameters.steps))
            plt.legend(loc='upper left')
            plt.xlabel("energy")
            if (parameters.hubbard_interaction == 0):
                plt.ylabel("Noninteracting green Function")
            else:
                plt.ylabel("Interacting green Function")
            plt.show()
    time_elapsed = (time.perf_counter() - time_start)
    print(" The time it took the computation is", time_elapsed)

    # print(h.heap())


if __name__ == "__main__":  # this will only run if it is a script and not a import module
    main()
    
*/