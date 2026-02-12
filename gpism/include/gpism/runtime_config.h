#pragma once

#include <map>
#include <string>

namespace gpism {

class RuntimeConfig {
public:
  RuntimeConfig();

  void load_defaults();
  bool load_file(const std::string& path, bool replace);
  bool apply_override(const std::string& path);

  bool has(const std::string& key) const;
  std::string get_string(const std::string& key) const;
  int get_int(const std::string& key) const;
  double get_double(const std::string& key) const;
  bool get_bool(const std::string& key) const;
  const std::string& last_error() const;

  void set(const std::string& key, const std::string& value);
  std::string summary() const;

private:
  std::map<std::string, std::string> values_;
  std::string last_error_;
};

}  // namespace gpism
