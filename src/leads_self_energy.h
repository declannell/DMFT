#pragma once
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <math.h>

typedef std::complex<double> dcomp;

class EmbeddingSelfEnergy
{
private:
    double kx_value, ky_value;
public:
    std::vector<dcomp> self_energy_left;
    std::vector<dcomp> self_energy_right;
    EmbeddingSelfEnergy(Parameters &parameters, double _kx, double _ky, int voltage_step);

    void get_transfer_matrix(Parameters &parameters, std::vector<dcomp> &transfer_matrix_l, std::vector<dcomp> &transfer_matrix_r, int voltage_step);
    void get_self_energy(Parameters &parameters, std::vector<dcomp> &transfer_matrix_l, std::vector<dcomp> &transfer_matrix_r, int voltage_step);
    double kx() const;
    double ky() const;
};

void run(Parameters &parameters);
std::vector<dcomp> analytic_self_energy(Parameters &parameters, int voltage_step);