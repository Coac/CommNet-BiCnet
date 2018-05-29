from concurrent import futures
from multiprocessing import cpu_count
import train_comm_net

def start_process():
    process = pool.submit(train_comm_net.main)
    process.arg = "lol"
    process.add_done_callback()
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
    print('Initializing Process Pool - {0} workers'.format(num_workers))
    pool = futures.ProcessPoolExecutor(max_workers=num_workers)
    for i in range(num_workers):
        start_process()
