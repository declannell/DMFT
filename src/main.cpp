#include <mpi.h>

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

#include "dmft.h"
#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"
#include "transport.h"

int main() {
  Parameters parameters = Parameters::init();

  std::vector<double> kx(parameters.chain_length_x, 0);
  std::vector<double> ky(parameters.chain_length_y, 0);

  for (int i = 0; i < parameters.chain_length_x; i++) {
    if (parameters.chain_length_x != 1) {
      kx.at(i) = 2 * M_PI * i / parameters.chain_length_x;
    } else if (parameters.chain_length_x == 1) {
      kx.at(i) = M_PI / 2.0;
    }
  }

  for (int i = 0; i < parameters.chain_length_y; i++) {
    if (parameters.chain_length_y != 1) {
      ky.at(i) = 2 * M_PI * i / parameters.chain_length_y;
    } else if (parameters.chain_length_y == 1) {
      ky.at(i) = M_PI / 2.0;
    }
  }

  std::vector<dcomp> current_up(parameters.NIV_points, 0);
  std::vector<dcomp> current_down(parameters.NIV_points, 0);
  std::vector<dcomp> current_up_right(parameters.NIV_points, 0);
  std::vector<dcomp> current_up_left(parameters.NIV_points, 0);     
  std::vector<dcomp> current_down_right(parameters.NIV_points, 0);
  std::vector<dcomp> current_down_left(parameters.NIV_points, 0);



  std::vector<Eigen::MatrixXcd> gf_local_up(
      parameters.steps,
      Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
  std::vector<Eigen::MatrixXcd> gf_local_down(
      parameters.steps,
      Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
  std::vector<Eigen::MatrixXcd> gf_local_lesser_up(
      parameters.steps,
      Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
  std::vector<Eigen::MatrixXcd> gf_local_lesser_up_FD(
      parameters.steps,
      Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
  std::vector<Eigen::MatrixXcd> gf_local_lesser_down(
      parameters.steps,
      Eigen::MatrixXcd::Zero(parameters.chain_length, parameters.chain_length));
  std::vector<std::vector<dcomp>> self_energy_mb_up(
      parameters.chain_length, std::vector<dcomp>(parameters.steps)),
      self_energy_mb_lesser_up(parameters.chain_length,
                               std::vector<dcomp>(parameters.steps, 0));
  std::vector<std::vector<dcomp>> self_energy_mb_down(
      parameters.chain_length, std::vector<dcomp>(parameters.steps)),
      self_energy_mb_lesser_down(parameters.chain_length,
                                 std::vector<dcomp>(parameters.steps, 0));

  for (int m = 0; m < parameters.NIV_points; m++) {
    std::cout << "The voltage difference is "
              << parameters.voltage_l[m] - parameters.voltage_r[m] << std::endl;

    std::vector<std::vector<EmbeddingSelfEnergy>> leads;
    for (int i = 0; i < parameters.chain_length_x; i++) {
      std::vector<EmbeddingSelfEnergy> vy;
      for (int j = 0; j < parameters.chain_length_y; j++) {
        vy.push_back(EmbeddingSelfEnergy(parameters, kx.at(i), ky.at(j), m));
      }
      leads.push_back(vy);
    }

    std::cout << "leads complete" << std::endl;
    /*
get_local_gf(parameters, kx, ky, self_energy_mb_up, leads, gf_local_up, m);
get_local_gf(parameters, kx, ky, self_energy_mb_down, leads, gf_local_down, m);
*/
    get_local_gf_r_and_lesser(parameters, kx, ky, self_energy_mb_up,
                              self_energy_mb_lesser_up, leads, gf_local_up,
                              gf_local_lesser_up, m);
    get_local_gf_r_and_lesser(parameters, kx, ky, self_energy_mb_down,
                              self_energy_mb_lesser_down, leads, gf_local_down,
                              gf_local_lesser_down, m);

    for (int r = 0; r < parameters.steps; r++) {
      for (int i = 0; i < parameters.chain_length; i++) {
        for (int j = 0; j < parameters.chain_length; j++) {
          gf_local_lesser_up_FD.at(r)(i, j) =
              -1.0 *
              fermi_function(parameters.energy.at(r).real(), parameters) *
              (gf_local_up.at(r)(i, j) - std::conj(gf_local_up.at(r)(j, i)));
        }
      }
    }

    double difference;
    if (m == 0) {
      get_difference(parameters, gf_local_lesser_up, gf_local_lesser_up_FD,
                     difference);
      std::cout << "The difference between the fD and other is " << difference
                << std::endl;
      std::cout << "got local green function" << std::endl;
    }

    std::vector<double> spins_occup(2 * parameters.chain_length);

    dmft(parameters, m, kx, ky, self_energy_mb_up, self_energy_mb_down,
         self_energy_mb_lesser_up, self_energy_mb_lesser_down, gf_local_up,
         gf_local_down, gf_local_lesser_up, gf_local_lesser_down, leads,
         spins_occup);

    std::cout << "got self energy" << std::endl;

    if (parameters.hubbard_interaction == 0 && parameters.chain_length == 1 &&
        parameters.chain_length_x == 1 && m == 0) {
      get_analytic_gf_1_site(parameters, gf_local_up, m);
    }

    std::vector<dcomp> transmission_up(parameters.steps, 0);
    std::vector<dcomp> transmission_down(parameters.steps, 0);
    if (parameters.hubbard_interaction == 0) {
      get_transmission(parameters, kx, ky, self_energy_mb_up, leads,
                       transmission_up, transmission_down, m);
      get_landauer_buttiker_current(parameters, transmission_up,
                                    transmission_down, &current_up.at(m),
                                    &current_down.at(m), m);

      get_meir_wingreen_current(
          parameters, kx, ky, self_energy_mb_up, self_energy_mb_lesser_up,
          leads, m, &current_up_left.at(m), &current_up_right.at(m));

      get_meir_wingreen_current(
          parameters, kx, ky, self_energy_mb_down, self_energy_mb_lesser_down,
          leads, m, &current_down_left.at(m), &current_down_right.at(m));
    } else {

      get_meir_wingreen_current(
          parameters, kx, ky, self_energy_mb_up, self_energy_mb_lesser_up,
          leads, m, &current_up_left.at(m), &current_up_right.at(m));
      get_meir_wingreen_current(
          parameters, kx, ky, self_energy_mb_down, self_energy_mb_lesser_down,
          leads, m, &current_down_left.at(m), &current_down_right.at(m));
    }

    /*
std::cout << "kx is " << std::endl;
for (int i = 0; i < parameters.chain_length_x; i++) {
std::cout << kx.at(i) << ",";
}
std::cout << "\n" << "ky is " << std::endl;
for (int i = 0; i < parameters.chain_length_y; i++) {
std::cout << ky.at(i) << "," << "\n";
}
*/

    std::ofstream gf_local_file;
    gf_local_file.open(
        "/home/declan/green_function_code/quantum_transport/textfiles/"
        "gf_c++.txt");
    // myfile << parameters.steps << std::endl;
    for (int i = 0; i < parameters.chain_length; i++) {
      for (int r = 0; r < parameters.steps; r++) {
        gf_local_file << parameters.energy.at(r).real() << ","
                      << gf_local_up.at(r)(i, i).real() << ","
                      << gf_local_up.at(r)(i, i).imag() << ","
                      << gf_local_down.at(r)(i, i).real() << ","
                      << gf_local_down.at(r)(i, i).imag() << "\n";

        // std::cout << leads.self_energy_left.at(r) << "\n";
      }
    }
    gf_local_file.close();

    std::ofstream transmission_file;
    transmission_file.open(
        "/home/declan/green_function_code/quantum_transport/textfiles/"
        "c++_tranmission.txt");
    // myfile << parameters.steps << std::endl;
    for (int r = 0; r < parameters.steps; r++) {
      transmission_file << parameters.energy.at(r).real() << ","
                        << transmission_up.at(r).real() << ","
                        << transmission_up.at(r).imag() << ","
                        << transmission_down.at(r).real() << "\n";
    }
    transmission_file.close();

    std::ofstream se_r_file;
    se_r_file.open(
        "/home/declan/green_function_code/quantum_transport/textfiles/"
        "c++se_mb_r.txt");
    for (int i = 0; i < parameters.chain_length; i++) {
      for (int r = 0; r < parameters.steps; r++) {
        se_r_file << parameters.energy.at(r).real() << ","
                  << self_energy_mb_up.at(i).at(r).real() << ","
                  << self_energy_mb_up.at(i).at(r).imag() << ","
                  << self_energy_mb_down.at(i).at(r).real() << ","
                  << self_energy_mb_down.at(i).at(r).imag() << "\n";
      }
    }



    std::ofstream se_lesser_file;
    se_lesser_file.open(
        "/home/declan/green_function_code/quantum_transport/textfiles/"
        "c++_se_mb_lesser.txt");
    for (int i = 0; i < parameters.chain_length; i++) {
      for (int r = 0; r < parameters.steps; r++) {
        se_lesser_file << parameters.energy.at(r).real() << ","
                       << self_energy_mb_lesser_up.at(i).at(r).real() << ","
                       << self_energy_mb_lesser_up.at(i).at(r).imag() << "\n";
      }
    }
    se_lesser_file.close();
  }
  if (parameters.hubbard_interaction == 0) {
    std::ofstream current_file;
    current_file.open(
        "/home/declan/green_function_code/quantum_transport/textfiles/"
        "c++_current.txt");
    // myfile << parameters.steps << std::endl;
    for (int m = 0; m < parameters.NIV_points; m++) {
      std::cout << "The spin up current is " << current_up.at(m)
                << "The spin down current is " << current_down.at(m) << "\n";

      std::cout << "The spin up left current is " << current_up_left.at(m)
                << "The spin up right current is " << current_up_right.at(m)
                << "The spin down left current is " << current_down_left.at(m)
                << "The spin up right current is " << current_down_right.at(m)
                << "\n";

      std::cout << "\n";
      current_file << parameters.voltage_l[m] - parameters.voltage_r[m] << ","
                   << current_up.at(m).real() << ","
                   << current_down.at(m).real() << ","
                   << current_up_left.at(m).real() << ","
                   << current_up_right.at(m).real() << ","
                   << current_down_left.at(m).real() << ","
                   << current_down_right.at(m).real() << "\n";
    }
    current_file.close();
  } else {

   std::ofstream current_file;
    current_file.open(
        "/home/declan/green_function_code/quantum_transport/textfiles/"
        "c++_current.txt");
    // myfile << parameters.steps << std::endl;
    for (int m = 0; m < parameters.NIV_points; m++) {
      std::cout << "The spin up left current is " << current_up_left.at(m)
                << "The spin up right current is " << current_up_right.at(m)
                << "The spin down left current is " << current_down_left.at(m)
                << "The spin up right current is " << current_down_right.at(m)
                << "\n";

      std::cout << "\n";

      current_file << parameters.voltage_l[m] - parameters.voltage_r[m] << ","
                   << current_up_left.at(m).real() << ","
                   << current_up_right.at(m).real() << ","
                   << current_down_left.at(m).real() << ","
                   << current_down_right.at(m).real() << "\n";
    }
    current_file.close();
  }

  return 0;
}