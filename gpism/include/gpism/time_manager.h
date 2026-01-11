#pragma once

namespace gpism {

class TimeManager {
public:
  TimeManager(double start, double dt, double end, double output_interval);

  double time() const { return time_; }
  double dt() const { return dt_; }
  int step() const { return step_; }
  bool done() const { return time_ >= end_ - 1e-12; }
  bool should_output() const { return time_ >= next_output_ - 1e-12; }

  void mark_output();
  void advance();

private:
  double time_;
  double dt_;
  double end_;
  double output_interval_;
  double next_output_;
  int step_;
};

}  // namespace gpism
