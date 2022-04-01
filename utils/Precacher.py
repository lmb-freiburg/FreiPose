from threading import Thread, Lock
import time


class Precacher(object):
    """
        This thing can be helpful if you want to read data according to some index list ('key_list').
        Given one index of the key_list, f.e. key_list[0] there shoud be a read_fct that will retrieve the
        associated data from hard disk i.e. data_item = read_fct(key_list[0]).
        This precacher builds on the assumption that if you are currently at some position on the list its most likely
        that you want to see items in the vicinity of the current item next. Thats why it will try to load neighboring
        elements in an neighborhood of (-n_size, n_size) around the last queried element.
    """
    def __init__(self, key_list, read_fct, n_size=5):
        self._stop = False

        self._key_list = key_list  # list of keys we operate on
        self._key_idx = 0  # current key id (position in the list)
        self._data = dict()  # dictionary that will hold all data we read
        self._n_size = n_size  # neighborhood size we attempt to precache
        self._read = read_fct # function to read data

        # intialize thread and mutex
        self._lock = Lock()
        self._interrupt = False
        self.thread = Thread(target=self._thread_loop, args=())
        self.thread.daemon = True

    def start(self):
        # let thread run
        self._stop = False
        self.thread.start()
        return self

    def stop(self):
        # tell thread to stop and then wait for it to happen
        self._stop = True
        self.thread.join()

    def update_keylist(self, key_list):
        self._lock.acquire()
        self._key_list = key_list
        self._key_idx = 0
        self._data = dict()
        self._lock.release()

    def get_data(self, this_k):
        # check if current pointer is valid
        if this_k not in self._key_list:
            raise Exception('Invalid key given')

        # update the current pointer
        self._interrupt = True  # make reader thread stop if it is currently working on something
        self._lock.acquire() # because we signaled an interrupt we should not be stuck here for long (max time for reading one sample)
        self._key_idx = self._key_list.index(this_k)

        if this_k in self._data.keys():
            # if the data is there copy it
            data = self._data[this_k]

        else:
            # otherwise read it right away
            data = self._check_read(
                self._key_list.index(this_k)
            )

        self._interrupt = False
        self._lock.release()
        return data

    def _is_in_range(self, i):
        # check if a given index is in our current neighborhood or not
        j = abs(self._key_idx - i)
        if j <= self._n_size and 0 <= i < len(self._key_list):
            return True
        return False

    def _check_read(self, j):
        if not self._is_in_range(j):
            return None
        if self._key_list[j] in self._data.keys():
            # skip already existing entries
            return None
        self._data[self._key_list[j]] = self._read(self._key_list[j])
        return self._data[self._key_list[j]]

    def _thread_loop(self):
        while not self._stop:
            self._lock.acquire()
            # 1. Check if we want to delete something from the data dict
            rm_list = [k for k in self._data.keys() if not self._is_in_range(self._key_list.index(k))]
            for k in rm_list:
                # print('Deleting:', k)
                self._data.pop(k)

            # 2. Check if we want to add something
            for i in range(self._n_size+1):
                # read samples in case no interrupt happened
                if not self._interrupt:
                    self._check_read(self._key_idx + i)  # go up
                if not self._interrupt:
                    self._check_read(self._key_idx - i)  # go down

                if self._interrupt:
                #     print('Interrupt happened')
                    break

            self._lock.release()
            time.sleep(0.01)


if __name__ == '__main__':
    def _read(k):
        print('Adding: ', k)
        time.sleep(1.0)
        return 'data%s' % k

    tmp = Precacher(
        ['%d' % i for i in range(25)],
        # [],
        _read,
        n_size=2
    )

    tmp.start()
    print('started')
    print('Data at 2:', tmp.get_data('2'))
    time.sleep(5)
    # print('Data at 12:', tmp.get_data('12'))
    # time.sleep(5)
    # print('Data at 18:', tmp.get_data('18'))
    # time.sleep(5)
    print('Data at 23:', tmp.get_data('23'))
    time.sleep(5)
    tmp.stop()
    print('done')
