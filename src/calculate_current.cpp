#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
typedef std::complex<double> dcomp;

double fermi_function(const double energy)
{
	if (energy < 0.0) {
		return 1.0;
	} else {
		return 0.0;
	}
}

void get_landauer_buttiker_current(const std::vector<double>& energy, const std::vector<double>& transmission_up, const std::vector<double>& transmission_down, double* current_up,
    double* current_down, const double voltage)
{
	double delta_energy = energy.at(1) - energy.at(0);
	int number_energy_points = energy.size();
	for (int r = 0; r < number_energy_points; r++) {
		*current_up -= delta_energy * transmission_up.at(r) * (fermi_function(energy.at(r) + voltage) - fermi_function(energy.at(r) - voltage));

		*current_down -= delta_energy * transmission_down.at(r) * (fermi_function(energy.at(r) + voltage) - fermi_function(energy.at(r) - voltage));
	}
}

void read_transmission_from_textfile(std::vector<double>& transmission_up, std::vector<double>& transmission_down, std::vector<double>& energy, int& number_energy_points)
{
	transmission_up.resize(number_energy_points);
	transmission_down.resize(number_energy_points);
	energy.resize(number_energy_points);
	fstream my_file;
	my_file.open("0.AuFeMgOFeAu.TRC", ios::in);
	int n = 2;

	for (int i = 0; i < n; i++) {
	}

	if (!my_file) {
		cout << "No such file\n";
	} else {
		int count = 0, index = 0;
		string line;

		while (1) {
			my_file >> line;
			index = std::floor(count / 11);
			if (my_file.eof()) {
				break;
			} else if (count % 11 == 0) {
				energy.at(index) = std::stod(line);
			} else if (count % 11 == 2) {
				transmission_up.at(index) = std::stod(line);
			} else if (count % 11 == 3) {
				transmission_down.at(index) += std::stod(line);
			}
			count++;
		}
	}
	my_file.close();
}

int main(int argc, char* argv[])
{
	//compile by g++ -std=c++11 src/calculate_current.cpp -o calculate_current
	std::vector<double> transmission_up;
	std::vector<double> transmission_down;
	std::vector<double> energy;

	if (argc != 3) {
		std::cout << "program doesn't have enough arguements.\n "
		          << " usage: executable  number_energy_points     bias" << std::endl;
		exit(1);
	}

	int number_energy_points = stoi(argv[1]) + 100;  //I add 100 energy point s cause smeagol is parallelised in a weird way.
	std::cout << "The numbe rof energy points is " << number_energy_points << std::endl;
	double voltage = atof(argv[2]);
	std::cout << "The numbe rof energy points is " << voltage << std::endl;

	read_transmission_from_textfile(transmission_up, transmission_down, energy, number_energy_points);

	//for(int r = 0; r < number_energy_points; r++){
	////    cout << transmission_up.at(r) << " " << transmission_down.at(r) << endl;
	//}

	double current_up, current_down;
	get_landauer_buttiker_current(energy, transmission_up, transmission_down, &current_up, &current_down, voltage);
	std::cout << voltage << " " << current_up << " " << current_down << "\n";
	return 0;
}