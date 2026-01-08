"""Static instance and pricing catalog for cost estimation and sizing."""

INSTANCE_CATALOG = {
    "c7i.4xlarge": {
        "family": "c7i",
        "vcpus": 16,
        "memory_mib": 32768,
        "gpus": 0,
        "on_demand_usd_per_hour": 0.68,
    },
    "hpc6a.48xlarge": {
        "family": "hpc6a",
        "vcpus": 192,
        "memory_mib": 393216,
        "gpus": 0,
        "on_demand_usd_per_hour": 2.88,
    },
    "g5.xlarge": {
        "family": "g5",
        "vcpus": 4,
        "memory_mib": 16384,
        "gpus": 1,
        "on_demand_usd_per_hour": 1.01,
    },
}

INSTANCE_ALIASES = {
    "c7i": "c7i.4xlarge",
    "hpc6a": "hpc6a.48xlarge",
    "g5": "g5.xlarge",
}

CPU_FAMILIES = {"c7i", "hpc6a"}
GPU_FAMILIES = {"g5"}
