# external
from sys import argv
import numpy as np
from multiprocessing import Pool
# local
from configure_setup import setup
from run_ts import parallel_worker

def run_TS_parallel(setup_config):
    """
    Splits requested input parameters into N_CPU chunks and calls
    parallel_worker N_CPU times in parallel with respective input

    Parameters
    ----------
    setup_config : setup
        Configuration of requested computations
    """

    if setup_config.ncpu > setup_config.inputParams['count']:
        setup_config.ncpu = setup_config.inputParams['count']
        print(f"Requested more CPUs than jobs. Will use {setup_config.ncpu} CPUs instead")

    ind = np.arange(setup_config.inputParams['count'])
    args = [ [setup_config, ind[i::setup_config.ncpu]] for i in range(setup_config.ncpu)]

    unpackFunc = lambda arg : parallel_worker((arg[0], arg[1]))
    with Pool(processes=setup_config.ncpu) as pool:
        pool.map(parallel_worker, args )


if __name__ == '__main__':
    try:
        conf_file = argv[1]
        setup_object = setup(file=conf_file)

        if len(argv) > 2:
            setup_object.jobID = argv[2]
        else:
            print("Usage: $ python generate_random_grid.py configFile.txt jobName")
            print("Assigning temporary job name: TMP")
            setup_object.jobID = 'TMP'

        run_TS_parallel(setup_object)

    except IndexError:
        print("Looking for input files ('config.txt' and 'input_param.txt') in ../input/ directory")
        #try:
        conf_file = "config.txt"
        print(conf_file)
        setup_object = setup(file=conf_file)
        setup_object.jobID = 'TMP'
        run_TS_parallel(setup_object)
        # except FileNotFoundError:
        #     print("Couldn't find 'config.txt' file in ../input/ dir. Place 'config.txt' file there or specify path:")
        #     print("$ python generate_random_grid.py configFile.txt jobName")
