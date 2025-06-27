import sys
sys.path.append("../.")

import argparse
import pathlib
from lips.benchmark.powergridBenchmark import PowerGridBenchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_name", help="The name of the environemnt", default="l2rpn_case14_sandbox", required=False)
    parser.add_argument("-ntr", "--n_training_data", help="Number of training data", required=True, default=100, type=int)
    parser.add_argument("-nv", "--n_validation_data", help="Number of training data", required=True, default=100, type=int)
    parser.add_argument("-nte", "--n_test_data", help="Number of training data", required=True, default=100, type=int)
    parser.add_argument("-no", "--n_ood_data", help="Number of training data", required=True, default=100, type=int)

    args = parser.parse_args()
    
    # ENV_NAME = "l2rpn_case14_sandbox"
    # ENV_NAME = "l2rpn_neurips_2020_track1_small"
    ENV_NAME = args.env_name

    PATH = pathlib.Path().resolve().parent
    BENCH_CONFIG_PATH = PATH / "configs" / (ENV_NAME + ".ini")
    DATA_PATH = PATH / "Datasets" / ENV_NAME / "DC"
    LOG_PATH = PATH / "lips_logs.log"
    
    if not DATA_PATH.exists():
        DATA_PATH.mkdir(mode=511, parents=True)
        
    
    benchmark = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                   benchmark_name="Benchmark4",
                                   load_data_set=False,
                                   config_path=BENCH_CONFIG_PATH,
                                   log_path=LOG_PATH)
    
    benchmark.generate(nb_sample_train=int(args.n_training_data),
                       nb_sample_val=int(args.n_validation_data),
                       nb_sample_test=int(args.n_test_data),
                       nb_sample_test_ood_topo=int(args.n_ood_data),
                       do_store_physics=True,
                       is_dc=True
                      )