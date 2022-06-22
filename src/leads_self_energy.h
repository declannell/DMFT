#pragma once
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <math.h>
#include "parameters.h"

typedef std::complex<double> dcomp;

class EmbeddingSelfEnergy
{
private:
    double kx_value, ky_value;
public:
    std::vector<dcomp> self_energy_left;
    std::vector<dcomp> self_energy_right;
    std::vector<std::vector<double>> pdos;
    EmbeddingSelfEnergy(const Parameters &parameters, double _kx, double _ky, int voltage_step);

    void get_transfer_matrix(const Parameters &parameters, std::vector<dcomp> &transfer_matrix_l, std::vector<dcomp> &transfer_matrix_r, int voltage_step);
    void get_self_energy(const Parameters &parameters, std::vector<dcomp> &transfer_matrix_l, std::vector<dcomp> &transfer_matrix_r, int voltage_step);
    void add_left_insulating_layers(const Parameters &parameters, int voltage_step, std::vector<std::vector<double>> &pdos);
    void add_right_insulating_layers(const Parameters &parameters, int voltage_step, std::vector<std::vector<double>> &pdos);
    double kx() const;
    double ky() const;
};

void run(Parameters &parameters);
std::vector<dcomp> analytic_self_energy(const Parameters &parameters, int voltage_step);

void get_k_averaged_embedding_self_energy(const Parameters parameters, std::vector<std::vector<EmbeddingSelfEnergy>> &leads);

void get_spectral_embedding_self_energy(const Parameters parameters, std::vector<std::vector<EmbeddingSelfEnergy>> &leads, int m);

void average_pdos( const Parameters parameters, std::vector<std::vector<EmbeddingSelfEnergy>> &leads, int m);