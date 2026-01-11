#include "gpism/device_policy.h"

namespace gpism {

namespace {
bool device_enabled_flag = true;
}  // namespace

bool device_enabled() { return device_enabled_flag; }

void set_device_enabled(bool enabled) { device_enabled_flag = enabled; }

}  // namespace gpism
