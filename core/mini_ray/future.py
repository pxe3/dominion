from multiprocessing import Queue
from queue import Empty
import time

class Future:
    def __init__(self, result_queue: Queue):
        self.result_queue = result_queue
        self._result = None
        self._has_result = False

    def get(self, timeout=None):
        if self._has_result:
            return self._result
        else:
            self._result = self.result_queue.get(timeout=timeout) # raises empty if timeout expires
            self._has_result = True
            return self._result
    
    def ready(self):
        if self._has_result:
            return True
        try:
            self._result = self.result_queue.get(timeout=0)
            self._has_result = True
            return True
        except Empty:
            return False