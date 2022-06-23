#include <mpi.h>

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

#include "dmft.h"
#include "interacting_gf.h"
#include "leads_self_energy.h"
#include "parameters.h"
#include "transport.h"
#include <iomanip>
#include <sstream>

void print_parameters(Parameters &parameters) {
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
  std::cout << "hopping_ins_l_cor = " << parameters.hopping_ins_l_cor
            << std::endl;
  std::cout << "hopping_ins_r_cor = " << parameters.hopping_ins_r_cor
            << std::endl;
  std::cout << "num_cor = " << parameters.num_cor << std::endl;
  std::cout << "parameters.num_ins_left  =" << parameters.num_ins_left
            << std::endl;
  std::cout << "num_ins_right = " << parameters.num_ins_right << std::endl;
  std::cout << "num_ky_points = " << parameters.num_ky_points << std::endl;
  std::cout << "num_kx_points = " << parameters.num_kx_points << std::endl;
  std::cout << "chemical_potential = " << parameters.chemical_potential
            << std::endl;
  std::cout << "temperature = " << parameters.temperature << std::endl;
  std::cout << "e_upper_bound = " << parameters.e_upper_bound << std::endl;
  std::cout << "e_lower_bound = " << parameters.e_lower_bound << std::endl;
  std::cout << "hubbard_interaction = " << parameters.hubbard_interaction
            << std::endl;
  std::cout << "voltage_step = " << parameters.voltage_step << std::endl;
  std::cout << "self_consistent_steps = " << parameters.self_consistent_steps
            << std::endl;
  std::cout << "read_in_self_energy = " << parameters.read_in_self_energy
            << std::endl;
  std::cout << "NIV_points = " << parameters.NIV_points << std::endl;
  std::cout << "delta_v = " << parameters.delta_v << std::endl;
  std::cout << "delta_leads = " << parameters.delta_leads << std::endl;
  std::cout << "delta_gf = " << parameters.delta_gf << std::endl;
  std::cout << "leads_3d = " << parameters.leads_3d << std::endl;
  std::cout << "parameters.interaction_order = " << parameters.interaction_order
            << std::endl;
  std::cout << "parameters.steps = " << parameters.steps << std::endl;
  std::cout << "parameters.chain_length = " << parameters.chain_length
            << std::endl;
}

int main() {
  Parameters parameters = Parameters::init();
  print_parameters(parameters);
  std::vector<double> kx(parameters.num_kx_points, 0);
  std::vector<double> ky(parameters.num_ky_points, 0);

  for (int i = 0; i < parameters.num_kx_points; i++) {
    if (parameters.num_kx_points != 1) {
      kx.at(i) = 2 * M_PI * i / parameters.num_kx_points;
    } else if (parameters.num_kx_points == 1) {
      kx.at(i) = M_PI / 2.0;
    }
  }

  for (int i = 0; i < parameters.num_ky_points; i++) {
    if (parameters.num_ky_points != 1) {
      ky.at(i) = 2 * M_PI * i / parameters.num_ky_points;
    } else if (parameters.num_ky_points == 1) {
      ky.at(i) = M_PI / 2.0;
    }
  }

  std::vector<dcomp> current_up(parameters.NIV_points, 0);
  std::vector<dcomp> current_down(parameters.NIV_points, 0);
  std::vector<dcomp> current_up_right(parameters.NIV_points, 0);
  std::vector<dcomp> current_up_left(parameters.NIV_points, 0);
  std::vector<dcomp> current_down_right(parameters.NIV_points, 0);
  std::vector<dcomp> current_down_left(parameters.NIV_points, 0);

  std::vector<std::vector<Eigen::MatrixXd>> hamiltonian(
      parameters.num_kx_points,
      std::vector<Eigen::MatrixXd>(
          parameters.num_ky_points,
          Eigen::MatrixXd::Zero(parameters.num_cor, parameters.num_cor)));

  std::vector<Eigen::MatrixXcd> gf_local_up(
      parameters.steps,
      Eigen::MatrixXcd::Zero(parameters.num_cor, parameters.num_cor));
  std::vector<Eigen::MatrixXcd> gf_local_down(
      parameters.steps,
      Eigen::MatrixXcd::Zero(parameters.num_cor, parameters.num_cor));
  std::vector<Eigen::MatrixXcd> gf_local_lesser_up(
      parameters.steps,
      Eigen::MatrixXcd::Zero(parameters.num_cor, parameters.num_cor));
  std::vector<Eigen::MatrixXcd> gf_local_lesser_down(
      parameters.steps,
      Eigen::MatrixXcd::Zero(parameters.num_cor, parameters.num_cor));
  std::vector<std::vector<dcomp>> self_energy_mb_up(
      parameters.num_cor, std::vector<dcomp>(parameters.steps)),
      self_energy_mb_lesser_up(parameters.num_cor,
                               std::vector<dcomp>(parameters.steps, 0));
  std::vector<std::vector<dcomp>> self_energy_mb_down(
      parameters.num_cor, std::vector<dcomp>(parameters.steps)),
      self_energy_mb_lesser_down(parameters.num_cor,
                                 std::vector<dcomp>(parameters.steps, 0));

  for (int m = 1; m < parameters.NIV_points; m++) {
    if (m != 0 &&
        parameters.leads_3d == true) { // this has already been initialised for
                                       // the equilibrium case in line 19-33

      kx.resize(parameters.num_kx_points);
      ky.resize(parameters.num_ky_points);
      for (int i = 0; i < parameters.num_kx_points; i++) {
        if (parameters.num_kx_points != 1) {
          kx.at(i) = 2 * M_PI * i / parameters.num_kx_points;
        } else if (parameters.num_kx_points == 1) {
          kx.at(i) = M_PI / 2.0;
        }
      }

      for (int i = 0; i < parameters.num_ky_points; i++) {
        if (parameters.num_ky_points != 1) {
          ky.at(i) = 2 * M_PI * i / parameters.num_ky_points;
        } else if (parameters.num_ky_points == 1) {
          ky.at(i) = M_PI / 2.0;
        }
      }
    }
    std::cout << "\n";
    std::cout << std::setprecision(15) << "The voltage difference is "
              << parameters.voltage_l[m] - parameters.voltage_r[m] << std::endl;

    std::vector<std::vector<EmbeddingSelfEnergy>> leads;
    for (int i = 0; i < parameters.num_kx_points; i++) {
      std::vector<EmbeddingSelfEnergy> vy;
      for (int j = 0; j < parameters.num_ky_points; j++) {
        vy.push_back(EmbeddingSelfEnergy(parameters, kx.at(i), ky.at(j), m));
      }
      leads.push_back(vy);
    }

    if ((parameters.num_ins_left != 0 or parameters.num_ins_right != 0)){
      average_pdos(parameters, leads, m);
    }

    if (parameters.leads_3d == true) {
      get_k_averaged_embedding_self_energy(parameters, leads);
      kx.resize(1);
      ky.resize(1);
      kx.at(0) = M_PI / 2.0;
      ky.at(0) = M_PI / 2.0;
    }

    for (int kx_i = 0; kx_i < parameters.num_kx_points; kx_i++) {
      for (int ky_i = 0; ky_i < parameters.num_ky_points; ky_i++) {
        get_hamiltonian(parameters, m, kx.at(kx_i), ky.at(ky_i),
                        hamiltonian.at(kx_i).at(ky_i));
        // std::cout << "The hamiltonian is " <<  std::endl;
        // std::cout << hamiltonian.at(kx_i).at(ky_i) << std::endl;
        // std::cout << std::endl;
      }
    }
    get_spectral_embedding_self_energy(parameters, leads, m);

    std::cout << "leads complete" << std::endl;
    get_local_gf_r_and_lesser(parameters, self_energy_mb_up,
                              self_energy_mb_lesser_up, leads, gf_local_up,
                              gf_local_lesser_up, m, hamiltonian);
    get_local_gf_r_and_lesser(parameters, self_energy_mb_down,
                              self_energy_mb_lesser_down, leads, gf_local_down,
                              gf_local_lesser_down, m, hamiltonian);

    // std::vector<Eigen::MatrixXcd> gf_local_lesser_up_FD(
    //    parameters.steps,
    //    Eigen::MatrixXcd::Zero(parameters.num_cor,
    //    parameters.num_cor));
    /// for (int r = 0; r < parameters.steps; r++) {
    ///  for (int i = 0; i < parameters.num_cor; i++) {
    ///    for (int j = 0; j < parameters.num_cor; j++) {
    ///      gf_local_lesser_up_FD.at(r)(i, j) =
    ///          -1.0 *
    ///          fermi_function(parameters.energy.at(r), parameters) *
    ///          (gf_local_up.at(r)(i, j) - std::conj(gf_local_up.at(r)(j, i)));
    ///    }
    ///  }
    ///}

    // double difference;
    // int index;
    // if (m == 0) {
    //  get_difference(parameters, gf_local_lesser_up, gf_local_lesser_up_FD,
    //                 difference, index);
    //  std::cout << "The difference between the fD and other is " << difference
    //            << std::endl;
    //  std::cout << "The index is " << index << std::endl;
    //  std::cout << "got local green function" << std::endl;
    //}

    std::vector<double> spins_occup(2 * parameters.num_cor);

    dmft(parameters, m, self_energy_mb_up, self_energy_mb_down,
         self_energy_mb_lesser_up, self_energy_mb_lesser_down, gf_local_up,
         gf_local_down, gf_local_lesser_up, gf_local_lesser_down, leads,
         spins_occup, hamiltonian);

    std::cout << "got self energy " << std::endl;

    if (parameters.hubbard_interaction == 0 && parameters.num_cor == 1 &&
        parameters.num_kx_points == 1 && m == 0 &&
        parameters.num_ins_left == 0 && parameters.num_ins_right == 0) {
      get_analytic_gf_1_site(parameters, gf_local_up, m);
    }

    std::vector<dcomp> transmission_up(parameters.steps, 0);
    std::vector<dcomp> transmission_down(parameters.steps, 0);
    if (parameters.hubbard_interaction == 0) {

      get_transmission(parameters, self_energy_mb_up, leads, transmission_up,
                       transmission_down, m, hamiltonian);
      std::cout << "Got transmission " << std::endl;
      get_landauer_buttiker_current(parameters, transmission_up,
                                    transmission_down, &current_up.at(m),
                                    &current_down.at(m), m);

      //get_meir_wingreen_current(
      //    parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m,
      //    &current_up_left.at(m), &current_up_right.at(m), hamiltonian);
//
      //get_meir_wingreen_current(
      //    parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, m,
      //    &current_down_left.at(m), &current_down_right.at(m), hamiltonian);
    } else {

      get_meir_wingreen_current(
          parameters, self_energy_mb_up, self_energy_mb_lesser_up, leads, m,
          &current_up_left.at(m), &current_up_right.at(m), hamiltonian);
      get_meir_wingreen_current(
          parameters, self_energy_mb_down, self_energy_mb_lesser_down, leads, m,
          &current_down_left.at(m), &current_down_right.at(m), hamiltonian);
      std::cout << "Got current " << std::endl;
    }

    for (int i = 0; i < parameters.num_cor; i++) {
      std::ostringstream ossgf;
      ossgf << "textfiles/" << m << "." << i + parameters.num_ins_left
            << ".gf.txt";
      std::string var = ossgf.str();

      std::ofstream gf_local_file;
      gf_local_file.open(var);
      for (int r = 0; r < parameters.steps; r++) {
        gf_local_file << parameters.energy.at(r) << "  "
                      << gf_local_up.at(r)(i, i).real() << "   "
                      << gf_local_up.at(r)(i, i).imag() << "   "
                      << gf_local_down.at(r)(i, i).real() << "   "
                      << gf_local_down.at(r)(i, i).imag() << " \n";

        // std::cout << leads.self_energy_left.at(r) << "\n";
      }
      gf_local_file.close();
    }



    for (int i = 0; i < parameters.num_cor; i++) {
      std::ostringstream ossgf;
      ossgf << "textfiles/" << m << "." << i + parameters.num_ins_left
            << ".gf_lesser.txt";
      std::string var = ossgf.str();

      std::ofstream gf_lesser_file;
      gf_lesser_file.open(var);
      for (int r = 0; r < parameters.steps; r++) {
        gf_lesser_file << parameters.energy.at(r) << "  "
                       << gf_local_lesser_up.at(r)(i, i).real() << "   "
                       << gf_local_lesser_up.at(r)(i, i).imag() << "   "
                       << gf_local_lesser_down.at(r)(i, i).real() << "   "
                       << gf_local_lesser_down.at(r)(i, i).imag() << " "
                       << -2.0 * parameters.j1 *
                              fermi_function(parameters.energy.at(r),
                                             parameters) *
                              gf_local_down.at(r)(i, i).imag()
                       << "\n";
      }
      gf_lesser_file.close();
    }

    std::ostringstream oss;
    oss << "textfiles/" << m << ".tranmission.txt";
    std::string var = oss.str();

    std::ofstream transmission_file;
    transmission_file.open(var);
    // myfile << parameters.steps << std::endl;
    for (int r = 0; r < parameters.steps; r++) {
      transmission_file << parameters.energy.at(r) << "  "
                        << transmission_up.at(r).real() << "  "
                        << transmission_down.at(r).real() << "\n";
    }
    transmission_file.close();

    std::ostringstream oss1;
    oss1 << "textfiles/" << m << ".dos.txt";
    var = oss1.str();

    std::ofstream dos_file;
    dos_file.open(var);
    // myfile << parameters.steps << std::endl;
    for (int r = 0; r < parameters.steps; r++) {
      double dos_total_up = 0.0;
      double dos_total_down = 0.0;
      for (int i = 0; i < parameters.num_cor; i++) {
        dos_total_up += -gf_local_up.at(r)(i, i).imag();
        dos_total_down += -gf_local_down.at(r)(i, i).imag();
      }
      dos_file << parameters.energy.at(r) << "  " << dos_total_up << "   "
               << dos_total_down << " \n";
    }
    dos_file.close();

    for (int i = 0; i < parameters.num_cor; i++) {
      std::ostringstream ossser;
      ossser << "textfiles/" << m << "." << i << ".se_r.txt";
      var = ossser.str();
      std::ofstream se_r_file;
      se_r_file.open(var);
      for (int r = 0; r < parameters.steps; r++) {
        se_r_file << parameters.energy.at(r) << "  "
                  << self_energy_mb_up.at(i).at(r).real() << "  "
                  << self_energy_mb_up.at(i).at(r).imag() << "  "
                  << self_energy_mb_down.at(i).at(r).real() << "  "
                  << self_energy_mb_down.at(i).at(r).imag() << "\n";
      }
      se_r_file.close();
    }

    for (int i = 0; i < parameters.num_cor; i++) {
      std::ostringstream osssel;
      osssel << "textfiles/" << m << "." << i + parameters.num_ins_left
             << ".se_l.txt";
      var = osssel.str();
      std::ofstream se_lesser_file;
      se_lesser_file.open(var);
      for (int r = 0; r < parameters.steps; r++) {
        dcomp x = -2.0 * fermi_function(parameters.energy.at(r), parameters) *
                  (self_energy_mb_up.at(i).at(r).imag());
        se_lesser_file << parameters.energy.at(r) << "  "
                       << self_energy_mb_lesser_up.at(i).at(r).real() << " "
                       << self_energy_mb_lesser_up.at(i).at(r).imag() << " "
                       << x.imag() << "\n";
      }
      se_lesser_file.close();
    }
  }

  if (parameters.hubbard_interaction == 0) {
    std::ofstream current_file;
    current_file.open("textfiles/"
                      "current.txt");
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
      current_file << parameters.voltage_l[m] - parameters.voltage_r[m]
                   << "   "
                   //<< current_up.at(m).real() << "   "
                   //<< current_down.at(m).real() << "   "
                   << 2 * current_up_left.at(m).real() << "   "
                   << 2 * current_up_right.at(m).real()
                   << "   "
                   //<< current_down_left.at(m).real() << "   "
                   << 2 * current_up_left.at(m).real() +
                          2 * current_up_right.at(m).real()
                   << "\n";
    }
    current_file.close();
  } else {

    std::ofstream current_file;
    current_file.open("textfiles/"
                      "current.txt");
    for (int m = 0; m < parameters.NIV_points; m++) {
      std::cout << "The spin up left current is " << current_up_left.at(m)
                << "The spin up right current is " << current_up_right.at(m)
                << "The spin down left current is " << current_down_left.at(m)
                << "The spin up right current is " << current_down_right.at(m)
                << "\n";

      std::cout << "\n";

      current_file << parameters.voltage_l[m] - parameters.voltage_r[m] << "   "
                   << current_up_left.at(m).real() << "   "
                   << current_up_right.at(m).real() << "   "
                   << current_down_left.at(m).real() << "   "
                   << current_down_right.at(m).real() << "     "
                   << current_down_left.at(m).real() +
                          current_down_right.at(m).real()
                   << "\n";
    }
    current_file.close();
  }

  return 0;
}