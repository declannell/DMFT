#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <complex>
using namespace std;
typedef std::complex<double> dcomp;

double kramer_kronig_relation(const std::vector<double> &enegry, std::vector<double>& impurity_self_energy_imag, int r);
void get_self_energy(const std::vector<dcomp>& gf_retarded, const std::vector<double> energy, int const number_energy_point, std::vector<dcomp>& impurity_self_energy);
void read_gf_from_textfile(std::vector<dcomp> &gf_retarded, std::vector<double> &energy, int &number_energy_points);
void get_dos(std::vector<double> &dos_up, std::vector<double> &dos_down, std::vector<double> &energy, int number_energy_points);
void get_occupation(const std::vector<double> &dos_up, const std::vector<double> &dos_down, const std::vector<double> &energy, 
	const int number_energy_points, double *occupation_up, double *occupation_down);
double fermi_function(const double energy);
void write_dos_to_file(const std::vector<double> &dos_up, const std::vector<double> &dos_down, const std::vector<double> &energy, 
	const int number_energy_points);


double kramer_kronig_relation(const std::vector<double> &energy, std::vector<double>& impurity_self_energy_imag, int r)
{
	double real_self_energy = 0;
	int number_energy_points = energy.size();

    double delta_energy = (energy.at(number_energy_points - 1) - energy.at(0)) / (double)number_energy_points;
	for (int q = 0; q < number_energy_points; q++) {
		if (q != r) {
			real_self_energy += delta_energy * impurity_self_energy_imag.at(q) / (energy.at(q) - energy.at(r));
        }
    }
	return real_self_energy / M_PI;
}


double fermi_function(const double energy) {
        if(energy < 0.0){
            return 1.0;
        } else {
            return 0.0;
        }
}

double integrate_equilibrium(std::vector<double>& gf_1, std::vector<double>& gf_2, std::vector<double>& gf_3, const std::vector<double> energy, int r)
{
	int number_energy_points = energy.size();

    double delta_energy = (energy.at(number_energy_points - 1) - energy.at(0)) / (double)number_energy_points;
	double result = 0;
	for (int i = 0; i < number_energy_points; i++) {
		for (int j = 0; j < number_energy_points; j++) {
			if (((i + j - r) > 0) && ((i + j - r) < number_energy_points)) {
				double prefactor = fermi_function(energy.at(i)) * fermi_function(energy.at(j)) 
					+ (1 - fermi_function(energy.at(i)) - fermi_function(energy.at(j))) 
					* fermi_function(energy.at(j) + energy.at(i) - energy.at(r)); 
				result += prefactor * (delta_energy / (M_PI)) * (delta_energy / (M_PI)) * gf_1.at(i) * gf_2.at(j) * gf_3.at(i + j - r);
			}
		}
	}
	return result;
}

void get_dos(std::vector<double> &dos_up, std::vector<double> &dos_down, std::vector<double> &energy, int number_energy_points) {

	fstream my_file;
	std::cout << "seg fault here \n";
	for (int i = 1; i < 4; i++){
		std::ostringstream osssel;
		osssel << "/home/declan/Downloads/0.TaFeMgOFeTa.TRC.EMPDOS.Fe_left_" << i << "_d_orbitals";
		
		std::string var = osssel.str();
		cout << var << endl;
		my_file.open(var, ios::in);
		if (!my_file) {
			cout << "No such file";
		} else {
			int count = 0, index = 0;
			string line;
			while (1) {
				my_file >> line;
				if (line.length() == 0 || line[0] == '#'){
					my_file.ignore(256,'\n');
				} else {
					//cout << line << "\n";
					index = std::floor(count / 3);
					if (my_file.eof()){
						break;
					} else if (count % 3 == 0) {
						energy.at(index) = std::stod(line);
					} else if (count % 3 == 1) {
						dos_up.at(index) += std::stod(line);
					} else if (count % 3 == 2) {
						dos_down.at(index) += std::stod(line);
					}
					count++;
				}
			}
		}
		my_file.close(); 
	}
	cout << "all done\n";
}


void get_self_energy(const std::vector<dcomp>& gf_retarded, const std::vector<double> energy, int const number_energy_point, std::vector<dcomp>& impurity_self_energy){

	std::vector<double> impurity_self_energy_real(number_energy_point), impurity_self_energy_imag(number_energy_point);
	std::vector<double> impurity_gf_up_imag(number_energy_point), impurity_gf_down_imag(number_energy_point);
	//std::vector<double> impurity_gf_up_real(parameters.steps), impurity_gf_down_real(parameters.steps);
	for (int r = 0; r < number_energy_point; r++) {
		impurity_gf_up_imag.at(r) = gf_retarded.at(r).imag();
		impurity_gf_down_imag.at(r) = gf_retarded.at(r).imag();
    }
    //I only want to calculate the imaginary part of the self energy.
    double hubbard_interaction =  0.1469972353;
	for (int r = 0; r < number_energy_point; r++){
		impurity_self_energy_imag.at(r) = hubbard_interaction * hubbard_interaction
		    * integrate_equilibrium(impurity_gf_up_imag, impurity_gf_down_imag, impurity_gf_down_imag, energy, r); 		
        std::cout << r << std::endl;
    }

    std::cout << "got the imag self energy" << endl;

	std::ostringstream ossser;
	ossser << "textfiles/"
	       << "se_krammer_kronig.txt";
	std::string var = ossser.str();
	std::ofstream se_krammer_kronig;
    dcomp j1 = -1;
    j1 = sqrt(j1);
	se_krammer_kronig.open(var);	
    for (int r = 0; r < number_energy_point; r++) {
		impurity_self_energy_real.at(r) = kramer_kronig_relation(energy, impurity_self_energy_imag, r);
		impurity_self_energy.at(r) = impurity_self_energy_real.at(r) + j1 * impurity_self_energy_imag.at(r);

		se_krammer_kronig << energy.at(r) << "  " << impurity_self_energy_real.at(r) << "  " << impurity_self_energy_imag.at(r) << "  " << impurity_self_energy.at(r)
		                  << "\n";
	}
	se_krammer_kronig.close();
}

void read_gf_from_textfile(std::vector<dcomp> &gf_retarded, std::vector<double> &energy, int &number_energy_points){
    number_energy_points = 4008;
    gf_retarded.resize(number_energy_points);
    energy.resize(number_energy_points);
	fstream my_file;
    dcomp j1 = -1;
    j1 = sqrt(j1);
	my_file.open("/home/declan/Downloads/Av-k_ReImGF_ReEne_K_1_1_1.dat", ios::in);
	if (!my_file) {
		cout << "No such file";
	} else {
        int count = 0, index = 0;
		string line;

		while (1) {
			my_file >> line;
            index = std::floor(count / 3);
			if (my_file.eof()){
				break;
            } else if (count % 3 == 0) {
                energy.at(index) = std::stod(line);
            } else if (count % 3 == 1) {
                gf_retarded.at(index) = std::stod(line);
            } else if (count % 3 == 2) {
                gf_retarded.at(index) += j1 * std::stod(line);
            }
            count++;
		}
	}
	my_file.close(); 
}

void get_occupation(const std::vector<double> &dos_up, const std::vector<double> &dos_down, const std::vector<double> &energy, 
	const int number_energy_points, double *occupation_up, double *occupation_down){
		double delta_energy = (energy.at(number_energy_points - 1) - energy.at(0)) / (double)number_energy_points;
		for (int r = 0; r < number_energy_points; r++) {
			*occupation_up += delta_energy * fermi_function(energy.at(r)) * dos_up.at(r);
			*occupation_down += delta_energy * fermi_function(energy.at(r)) * dos_down.at(r);
		}
}

void write_dos_to_file(const std::vector<double> &dos_up, const std::vector<double> &dos_down, const std::vector<double> &energy, 
	const int number_energy_points){
			std::ostringstream dos;
			dos << "textfiles/dos.txt";
			std::string var = dos.str();

			std::ofstream dos_file;
			dos_file.open(var);
			for (int r = 0; r < number_energy_points; r++) {
				dos_file << energy.at(r) << "  " << dos_up.at(r) << "   " << dos_down.at(r) << "\n";
			}
			dos_file.close();
	}

int main(){
	//compile by g++  -O3 -g  src/get_self_energy_from_file.cpp
    std::vector<dcomp> gf_retarded;
    std::vector<double> energy;
	std::vector<double> dos_up, dos_down;
    int number_energy_points = 528;
	double occupation_up, occupation_down;
	dos_up.resize(number_energy_points, 0);
	dos_down.resize(number_energy_points, 0);
	energy.resize(number_energy_points, 0);
    //read_gf_from_textfile(gf_retarded, energy, number_energy_points);
//
    //for(int r = 0; r < number_energy_points; r++){
    //    cout << gf_retarded.at(r) << endl;
    //}
    //cout << j1 << endl;
    //std::vector<dcomp> self_energy(number_energy_points);
    //get_self_energy(gf_retarded, energy, number_energy_points, self_energy);

	get_dos(dos_up, dos_down, energy, number_energy_points);
	get_occupation(dos_up, dos_down, energy, 
		number_energy_points, &occupation_up, &occupation_down);
	cout << "The spin up occupancy is " << occupation_up / 3.0 << std::endl;
	cout << "The spin dwon occupancy is " << occupation_down / 3.0 << std::endl;
	write_dos_to_file(dos_up, dos_down, energy, number_energy_points);


    return 0;
}

