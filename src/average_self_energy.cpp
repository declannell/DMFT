#include <vector>
#include<iostream>
#include<fstream>
#include <string>
#include <complex>
using namespace std;
typedef std::complex<double> dcomp;

void split_c(string str, double* real_self_energy, double* imag_self_energy)
{
    string w = "";
    for (auto x : str)
    {
        if (x == ',')
        { 
            *real_self_energy = std::stod(w);
            w = "";
        }
        else {
            w = w + x;
        }
    }
    *imag_self_energy = std::stod(w);
}


string sanitize(string word)
{
int i = 0;

while(i < word.size())
{
    if(word[i] == '(' || word[i] == ')')
    {
        word.erase(i,1);
    } else {
        i++;
    }
}
return word;
}

void average_self_energy(std::vector<dcomp> &self_energy_up, std::vector<dcomp> &self_energy_down, std::vector<double> &energy, int &number_energy_points){
    self_energy_up.resize(number_energy_points);
    self_energy_down.resize(number_energy_points);
    energy.resize(number_energy_points);
    double real, imag;
    int number_orbitals = 30;
    int number_of_lines = number_orbitals + 1;
    dcomp j1 = -1;
    j1 = sqrt(j1);
	fstream my_file;
	my_file.open("/home/declan/Downloads/SigmaMB.dat", ios::in);
	if (!my_file) {
		cout << "No such file";
	} else {
        int count = 0, index = 0;
		string line;
		while (1) {
			my_file >> line;
            index = std::floor(count / number_of_lines);
            if (index >= 2 * number_energy_points){
                break;
            }
            if (index >= number_energy_points) {
                index -= number_energy_points;
            }
            
            if (count % number_of_lines == 0){
                energy.at(index) = std::stod(line);
                //std::cout << "the energy.at(" << index << ") is " << energy.at(index) << std::endl;
            } else {
                line = sanitize(line);
                //std::cout << line << std::endl;
                split_c(line, &real, &imag);
                //std::cout << "The real part is still " << imag << std::endl;
                //std::cout << "Dividing by 30 gives " << imag/ (double)number_orbitals << std::endl;
                if (count > number_orbitals * number_energy_points) {
                    self_energy_down.at(index) += real / (double)number_orbitals;
                    self_energy_down.at(index) += j1 * imag / (double)number_orbitals;                    
                } else {
                    self_energy_up.at(index) += real / (double)number_orbitals;
                    self_energy_up.at(index) += j1 * imag / (double)number_orbitals;
                }         
            }
            

            count++;
            //cout << count << endl;
		}
	}
	my_file.close(); 
}

int main(){
    //compile by g++  -O3 -g  src/average_self_energy.cpp
    std::vector<dcomp> self_energy_up;
    std::vector<dcomp> self_energy_down;
    std::vector<double> energy;
    int number_energy_points = 3240;
    average_self_energy(self_energy_up, self_energy_down, energy, number_energy_points);

    std::ostringstream oss;
    oss << "/home/declan/Downloads/averaged_self_energy.txt";
    std::string var = oss.str();

    std::ofstream avergaded_self_energy_file;
    avergaded_self_energy_file.open(var);
    // myfile << parameters.steps << std::endl;
    for (int r = 0; r < number_energy_points; r++) {
        avergaded_self_energy_file << energy.at(r) << "  " << self_energy_up.at(r).real() << "  " << self_energy_up.at(r).imag() 
            << " " << self_energy_down.at(r).real() << "  " << self_energy_down.at(r).imag()
                            << "\n";
    }
    avergaded_self_energy_file.close();

    return 0;
}