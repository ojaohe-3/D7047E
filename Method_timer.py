import datetime
import time

def Method_timer(func):
    def timer(*args, **kwargs):
        print('calling {} method at {}'.format(func.__name__, datetime.datetime.now().strftime("%D: %H:%M:%S")))
        start = time.perf_counter()
        func(*args, **kwargs)
        total = time.perf_counter() - start
        print(f'finished {datetime.datetime.now().strftime("%D: %H:%M:%S")} after {total:0.4f}s')
    return timer