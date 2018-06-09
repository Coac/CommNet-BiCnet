from concurrent import futures
from multiprocessing import cpu_count
import train_comm_net
import itertools
import shlex

def start_process(args):
    process = pool.submit(train_comm_net.main, args)
    process.arg = args
    process.add_done_callback(done_callback)
    return False


def done_callback(process):
    if process.cancelled():
        print('Process {0} was cancelled'.format(process.arg))
    elif process.done():
        error = process.exception()
        if error:
            print('Process {0} - {1} '.format(process.arg, error))
        else:
            print('Process {0} done'.format(process.arg))


if __name__ == '__main__':
    num_workers = cpu_count()
    num_workers = 100

    print('Initializing Process Pool - {0} workers'.format(num_workers))
    pool = futures.ProcessPoolExecutor(max_workers=num_workers)

    params = {
        "--actor-lr": [0.01, 0.05, 0.1, 0.15],
        "--critic-lr": [0.01, 0.05, 0.1, 0.15]
    }


    hyperparams_names = list(params.keys())
    hyperparams = list(itertools.product(*params.values()))
    print("Number of run needed:", len(hyperparams))

    for hyperparam in hyperparams:
        args = ""
        for index, value in enumerate(hyperparam):
            args += hyperparams_names[index] + ' ' + str(value) + " "

        start_process(shlex.split(args))
