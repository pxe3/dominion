from multiprocessing import Process, Queue, Manager
from core.mini_ray.future import Future


class ActorHandle:
    def __init__(self, actor_class, *args, **kwargs):
        self.manager = Manager()
        self.request_queue = Queue()
        self.process = Process(
            target=ActorHandle._actor_loop,
            args=(actor_class, args, kwargs, self.request_queue)
        )
        self.process.start()
    
    def call(self, method_name, *args, **kwargs):
        result_queue = self.manager.Queue()
        self.request_queue.put((method_name, args, kwargs, result_queue))
        return Future(result_queue)

    @staticmethod
    def _actor_loop(actor_class, args, kwargs, request_queue):
        actor = actor_class(*args, **kwargs)
        while True:
            method_name, method_args, method_kwargs, result_queue = request_queue.get()
            if method_name is None:
                break
            method = getattr(actor, method_name)
            result = method(*method_args, **method_kwargs)
            result_queue.put(result)
    
    
    def shutdown(self):
        self.request_queue.put((None, None, None, None))
        self.process.join()


