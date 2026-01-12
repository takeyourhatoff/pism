#include "gpism/runtime_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace gpism {
namespace {

std::string trim(std::string value) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
              value.end());
  return value;
}

bool parse_line(const std::string& line, std::string* key, std::string* value) {
  const std::string stripped = trim(line);
  if (stripped.empty() || stripped[0] == '#') {
    return false;
  }
  const std::size_t eq = stripped.find('=');
  if (eq == std::string::npos) {
    return false;
  }
  *key = trim(stripped.substr(0, eq));
  *value = trim(stripped.substr(eq + 1));
  return !key->empty();
}

}  // namespace

RuntimeConfig::RuntimeConfig() { load_defaults(); }

void RuntimeConfig::load_defaults() {
  values_.clear();
  values_["grid.Mx"] = "100";
  values_["grid.My"] = "100";
  values_["grid.Mz"] = "20";
  values_["grid.Lz"] = "4000";
  values_["time.start_year"] = "0";
  values_["time.years"] = "1";
  values_["time.dt"] = "0.1";
  values_["time.output_interval"] = "1";
  values_["forcing.smb_constant"] = "0";
  values_["thermo.enabled"] = "0";
  values_["thermo.kappa"] = "1.0";
  values_["thermo.surface_value"] = "0.0";
  values_["thermo.basal_value"] = "0.0";
  values_["thermo.enthalpy_gamma"] = "0.0";
  values_["thermo.enthalpy_ref"] = "0.0";
  values_["thickness.evolve"] = "1";
  values_["ssa.enabled"] = "1";
  values_["ssa.max_picard"] = "10";
  values_["ssa.gmres_max_iter"] = "200";
  values_["ssa.tol_nuH"] = "1e-6";
  values_["ssa.tol_vel"] = "1e-6";
  values_["ssa.gmres_tol"] = "1e-8";
  values_["ssa.mg.enabled"] = "1";
  values_["ssa.mg.pre_iters"] = "2";
  values_["ssa.mg.post_iters"] = "2";
  values_["ssa.mg.coarse_iters"] = "10";
  values_["ssa.mg.omega"] = "1.0";
  values_["ssa.mg.min_size"] = "4";
  values_["ssa.mg.smoother"] = "jacobi";
  values_["ssa.mg.chebyshev.lambda_min"] = "0.1";
  values_["ssa.mg.chebyshev.lambda_max"] = "2.0";
  values_["ssa.mg.chebyshev.estimate"] = "0";
  values_["ssa.mg.chebyshev.estimate_iters"] = "5";
  values_["ssa.mg.chebyshev.estimate_min_factor"] = "0.1";
  values_["ssa.mg.chebyshev.estimate_max_factor"] = "1.1";
  values_["ssa.tauc_default"] = "2e5";
  values_["ssa.tauc_floor"] = "0.0";
  values_["ssa.mg.diagnostic"] = "0";
  values_["ssa.gmres.precond_diagnostic"] = "0";
  values_["io.format"] = "netcdf";
  values_["io.size"] = "small";
  values_["io.time_index"] = "-1";
}

bool RuntimeConfig::load_file(const std::string& path, bool replace) {
  if (replace) {
    values_.clear();
  }
  std::ifstream input(path);
  if (!input) {
    return false;
  }

  std::string line;
  while (std::getline(input, line)) {
    std::string key;
    std::string value;
    if (!parse_line(line, &key, &value)) {
      continue;
    }
    values_[key] = value;
  }
  return true;
}

bool RuntimeConfig::apply_override(const std::string& path) {
  return load_file(path, false);
}

bool RuntimeConfig::has(const std::string& key) const {
  return values_.find(key) != values_.end();
}

std::string RuntimeConfig::get_string(const std::string& key) const {
  auto it = values_.find(key);
  if (it == values_.end()) {
    throw std::runtime_error("Missing config key: " + key);
  }
  return it->second;
}

int RuntimeConfig::get_int(const std::string& key) const {
  return std::stoi(get_string(key));
}

double RuntimeConfig::get_double(const std::string& key) const {
  return std::stod(get_string(key));
}

bool RuntimeConfig::get_bool(const std::string& key) const {
  const std::string value = get_string(key);
  if (value == "1" || value == "true" || value == "TRUE") {
    return true;
  }
  if (value == "0" || value == "false" || value == "FALSE") {
    return false;
  }
  throw std::runtime_error("Invalid boolean config value for " + key);
}

void RuntimeConfig::set(const std::string& key, const std::string& value) {
  values_[key] = value;
}

std::string RuntimeConfig::summary() const {
  std::vector<std::string> keys;
  keys.reserve(values_.size());
  for (const auto& item : values_) {
    keys.push_back(item.first);
  }
  std::sort(keys.begin(), keys.end());

  std::ostringstream out;
  for (const auto& key : keys) {
    out << key << " = " << values_.at(key) << '\n';
  }
  return out.str();
}

}  // namespace gpism
