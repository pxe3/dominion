from core.mini_ray.actor import ActorHandle


class MethodProxy:
    def __init__(self, handle, method_name):
        self.handle = handle
        self.method_name = method_name
    
    def remote(self, *args, **kwargs):
        return self.handle.call(self.method_name, *args, **kwargs)
    
class RemoteHandle:
    def __init__(self, actor_handle):
        self._actor_handle = actor_handle

    def __getattr__(self, name):
        return MethodProxy(self._actor_handle, name)

    def shutdown(self):
        self._actor_handle.shutdown()


def remote(cls):
    def remote_init(*args, **kwargs):
        actor_handle = ActorHandle(cls, *args, **kwargs)
        return RemoteHandle(actor_handle)

    cls.remote = staticmethod(remote_init)
    return cls