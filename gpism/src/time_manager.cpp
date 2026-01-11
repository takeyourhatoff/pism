#include "gpism/time_manager.h"

#include <algorithm>

namespace gpism {

TimeManager::TimeManager(double start, double dt, double end,
                         double output_interval)
    : time_(start),
      dt_(dt > 0.0 ? dt : 1.0),
      end_(end),
      output_interval_(output_interval > 0.0 ? output_interval : dt_),
      next_output_(start),
      step_(0) {
  if (end_ < start) {
    end_ = start;
  }
}

void TimeManager::mark_output() { next_output_ += output_interval_; }

void TimeManager::advance() {
  time_ += dt_;
  ++step_;
  if (time_ > end_) {
    time_ = end_;
  }
}

}  // namespace gpism
