#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include "parameters.h"
#include "leads_self_energy.h"
// void function(vector<vector<vector<double>>> &green_function )

// this is a fake make file
// g++ -g -O -c parameters.cpp
// g++ -g -Wall  .\leads_self_energy.cpp -lm parameters.o
// g++ -g -Wall  main.cpp -lm parameters.o
double EmbeddingSelfEnergy::kx() const { return kx_value; }
double EmbeddingSelfEnergy::ky() const { return ky_value; }

EmbeddingSelfEnergy::EmbeddingSelfEnergy(const Parameters &parameters, double kx, double ky, int voltage_step) : kx_value(kx), ky_value(ky) // type is implied, it knows this is a constructor
{
    self_energy_left.resize(parameters.steps);
    self_energy_right.resize(parameters.steps);
    std::vector<dcomp> transfer_matrix_l(parameters.steps);
    std::vector<dcomp> transfer_matrix_r(parameters.steps);
    get_transfer_matrix(parameters, transfer_matrix_l, transfer_matrix_r, voltage_step);
    get_self_energy(parameters, transfer_matrix_l, transfer_matrix_r, voltage_step);
    // get_self_energy();
}

void EmbeddingSelfEnergy::get_transfer_matrix(const Parameters &parameters, std::vector<dcomp> &transfer_matrix_l, std::vector<dcomp> &transfer_matrix_r, int voltage_step)
{
    std::vector<dcomp> t_next_l(parameters.steps);
    std::vector<dcomp> t_next_r(parameters.steps);
    std::vector<dcomp> t_product_l(parameters.steps);
    std::vector<dcomp> t_product_r(parameters.steps);

    for (int r = 0; r < parameters.steps; r++)
    {
        t_next_l.at(r) = parameters.hopping_lz / (parameters.energy.at(r) + parameters.j1 * parameters.delta_leads - parameters.onsite_l - parameters.voltage_l[voltage_step] - 2 * parameters.hopping_ly * cos(this->ky()) - 2 * parameters.hopping_lx * cos(this->kx())); // dont need to do this-> but it looks like i can code
        t_next_r.at(r) = parameters.hopping_rz / (parameters.energy.at(r) + parameters.j1 * parameters.delta_leads - parameters.onsite_r - parameters.voltage_r[voltage_step] - 2 * parameters.hopping_ry * cos(this->ky()) - 2 * parameters.hopping_lx * cos(this->kx()));
        t_product_l.at(r) = t_next_l.at(r);
        t_product_r.at(r) = t_next_r.at(r);
        transfer_matrix_l.at(r) = t_next_l.at(r);
        transfer_matrix_r.at(r) = t_next_r.at(r);
    }

    std::vector<dcomp> old_transfer(parameters.steps, 0);

    double difference, real_difference, imag_difference;
    int count = 0;
    do
    {
        difference = -std::numeric_limits<double>::infinity();
        for (int r = 0; r < parameters.steps; r++)
        {
            t_next_l.at(r) = t_next_l.at(r) * t_next_l.at(r)/ (1.0 - 2.0 * pow(t_next_l.at(r), 2));
            t_next_r.at(r) = t_next_r.at(r) * t_next_r.at(r) / (1.0 - 2.0 * pow(t_next_r.at(r), 2));
            t_product_l.at(r) = t_product_l.at(r) * t_next_l.at(r);
            t_product_r.at(r) = t_product_r.at(r) * t_next_r.at(r);
            transfer_matrix_l.at(r) = transfer_matrix_l.at(r) + t_product_l.at(r);
            transfer_matrix_r.at(r) = transfer_matrix_r.at(r) + t_product_r.at(r);


            real_difference = abs(transfer_matrix_l.at(r).real() - old_transfer.at(r).real());
            imag_difference = abs(transfer_matrix_l.at(r).imag() - old_transfer.at(r).imag());
            difference = std::max(difference, std::max(real_difference, imag_difference));
            old_transfer.at(r) = transfer_matrix_l.at(r);
        }
        count++;

        //std::cout << "The difference is " << difference << "for a count of " << count << std::endl;

    } while (difference > 0.01 && count < 50);
}

void EmbeddingSelfEnergy::get_self_energy(const Parameters &parameters, std::vector<dcomp> &transfer_matrix_l, std::vector<dcomp> &transfer_matrix_r, int voltage_step)
{
    std::vector<dcomp> surface_gf_l(parameters.steps);
    std::vector<dcomp> surface_gf_r(parameters.steps);
    for (int r = 0; r < parameters.steps; r++)
    {
        surface_gf_l.at(r) = 1.0 / (parameters.energy.at(r) + parameters.j1 * parameters.delta_leads - parameters.voltage_l[voltage_step] - parameters.onsite_l - 2.0 * parameters.hopping_ly * cos(this->ky()) - 2.0 * parameters.hopping_lx * cos(this->kx()) - parameters.hopping_lz * transfer_matrix_l.at(r));
        surface_gf_r.at(r) = 1.0 / (parameters.energy.at(r) + parameters.j1 * parameters.delta_leads - parameters.voltage_r[voltage_step] - parameters.onsite_r - 2.0 * parameters.hopping_ry * cos(this->ky()) - 2.0 * parameters.hopping_lx * cos(this->kx()) - parameters.hopping_rz * transfer_matrix_r.at(r));
        this->self_energy_left.at(r) = parameters.hopping_lc * parameters.hopping_lc * surface_gf_l.at(r);
        this->self_energy_right.at(r) = parameters.hopping_rc * parameters.hopping_rc * surface_gf_r.at(r);
    }
}

template <typename T>
int sgn(T val)
{
    T zero_value = T(0);
    if (val == zero_value) {
        return 0;
    } else if (val > zero_value) {
        return 1;
    } else {
        return -1;
    }
}

std::vector<dcomp> analytic_self_energy(const Parameters &parameters, int voltage_step)
{

    std::vector<dcomp> analytic_se(parameters.steps);
    for (int r = 0; r < parameters.steps; r++)
    {
        double x = (parameters.energy.at(r) - parameters.onsite_l - parameters.voltage_l[voltage_step]) / (2.0 * parameters.hopping_lz);

        analytic_se.at(r) = parameters.hopping_lc * parameters.hopping_lc * (1.0 / abs(parameters.hopping_lz)) * (x + + parameters.j1 * parameters.delta_leads);
        if (abs(x) > 1.0)
        {
            analytic_se.at(r) = analytic_se.at(r).real() -parameters.hopping_lc * parameters.hopping_lc * (1.0 / abs(parameters.hopping_lz)) * (sgn(x) * sqrt(abs(x) * abs(x) - 1.0));
        }
        else if (abs(x) < 1.0)
        {
            analytic_se.at(r) = analytic_se.at(r).real() - parameters.j1 * parameters.hopping_lc * parameters.hopping_lc * abs((1.0 / abs(parameters.hopping_lz))) * (sqrt(1.0 - abs(x) * abs(x)));
        }
    }
    return analytic_se;
}

void run(const Parameters &parameters)
{
    EmbeddingSelfEnergy leads(parameters, M_PI / 2.0, M_PI / 2.0, parameters.voltage_step);

    std::ofstream myfile;
    myfile.open("textfiles/self_energy_lead_real.txt");
    // myfile << parameters.steps << std::endl;
    for (int r = 0; r < parameters.steps; r++)
    {
        myfile << parameters.energy.at(r) << "," << leads.self_energy_left.at(r).real() << "," << leads.self_energy_left.at(r).imag() << "," << leads.self_energy_right.at(r).real() << "," << leads.self_energy_right.at(r).imag() << ","
               << "\n";
        // std::cout << leads.self_energy_left.at(r) << "\n";
    }

    myfile.close();
}

void get_k_averaged_embedding_self_energy(const Parameters parameters, std::vector<std::vector<EmbeddingSelfEnergy>> &leads){
    int num_k = parameters.num_kx_points * parameters.num_ky_points;
    std::vector<dcomp> k_averaged_self_energy_left(parameters.steps), k_averaged_self_energy_right(parameters.steps);
    for( int kx_i = 0; kx_i < parameters.num_kx_points; kx_i++) {
        for( int ky_i = 0; ky_i < parameters.num_ky_points; ky_i++) {
            for(int r = 0; r < parameters.steps; r++) {
                k_averaged_self_energy_left.at(r) += leads.at(kx_i).at(ky_i).self_energy_left.at(r) / (double)num_k;
                k_averaged_self_energy_right.at(r) += leads.at(kx_i).at(ky_i).self_energy_right.at(r) / (double)num_k;
            }
        }
    }

    //std::cout << "core has not being dumped yet 1\n";
    //std::vector<EmbeddingSelfEnergy> vy;
    //vy.push_back(EmbeddingSelfEnergy(parameters, 0.0, 0.0, 0));
 //
//
    //leads.resize(1, vy);
    //for(int r = 0; r < parameters.steps; r++) {
    //    leads.at(0).at(0).self_energy_left.at(r) = k_averaged_self_energy_left.at(r);
    //    leads.at(0).at(0).self_energy_right.at(r) = k_averaged_self_energy_right.at(r);
    //}

    std::ofstream embedding_se_file;
    embedding_se_file.open(
        "textfiles/"
        "embedding_self_energy.txt");
    // myfile << parameters.steps << std::endl;

	for (int r = 0; r < parameters.steps; r++) {
		embedding_se_file << parameters.energy.at(r) << "  "
				<< k_averaged_self_energy_left.at(r).real() << "  "
				<< k_averaged_self_energy_left.at(r).imag() << "  "
				<< k_averaged_self_energy_right.at(r).real() << "  " 
                << k_averaged_self_energy_right.at(r).imag() <<"\n";
	}
	embedding_se_file.close();
}

void get_spectral_embedding_self_energy(const Parameters parameters, std::vector<std::vector<EmbeddingSelfEnergy>> &leads, int m){
    int num_k = parameters.num_kx_points * parameters.num_ky_points;
    std::vector<dcomp> k_averaged_self_energy_left(parameters.steps), k_averaged_self_energy_right(parameters.steps);
    for( int kx_i = 0; kx_i < parameters.num_kx_points; kx_i++) {
        for( int ky_i = 0; ky_i < parameters.num_ky_points; ky_i++) {
            for(int r = 0; r < parameters.steps; r++) {
                k_averaged_self_energy_left.at(r) += leads.at(kx_i).at(ky_i).self_energy_left.at(r) / (double)num_k;
                k_averaged_self_energy_right.at(r) += leads.at(kx_i).at(ky_i).self_energy_right.at(r) / (double)num_k;
            }
        }
    }

    std::ostringstream oss;
    oss << "textfiles/" << m << ".spectral_leads.txt";
    std::string var = oss.str();

    std::ofstream embedding_se_file;
    embedding_se_file.open(var);
    // myfile << parameters.steps << std::endl;

	for (int r = 0; r < parameters.steps; r++) {
		embedding_se_file << parameters.energy.at(r) << "  "
				<< k_averaged_self_energy_left.at(r).real() << "  "
				<< k_averaged_self_energy_left.at(r).imag() << "  "
				<< k_averaged_self_energy_right.at(r).real() << "  " 
                << k_averaged_self_energy_right.at(r).imag() <<"\n";
	}
	embedding_se_file.close();
}


