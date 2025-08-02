import threading
import queue
import time
from VAPWrapper import VAPWrapper

class VAPObjectPool:
    """Thread-safe object pool for VAP instances"""
    
    def __init__(self, configs: dict):
        self.size = configs['max_instance_cnt']
        self.configs = configs
        self.pool = queue.Queue(maxsize=self.size)
        self.lock = threading.Lock()
        self.created_count = 0
        
        # Pre-create VAP instances
        for i in range(self.size):
            vap_instance = VAPWrapper(
                model_path=configs['model_path'],
                context_size=configs['context_size_sec'],
                step_size=configs['step_size_sec'],
                frame_hz=configs['frame_hz'],
                device=configs['device'],
                debug_time=False,
            )
            self.pool.put(VAPPooledObject(vap_instance, self))
            self.created_count += 1
        
        print(f"VAP Object Pool initialized with {self.created_count} instances")
    
    def acquire(self, timeout=5.0):
        """Acquire a VAP instance from the pool"""
        try:
            obj = self.pool.get(timeout=timeout)
            obj.in_use = True  # Mark as in use
            return obj
        except queue.Empty:
            print("Failed to acquire VAP instance: pool is empty")
            return None
    
    def release(self, obj):
        """Release a VAP instance back to the pool"""
        if obj and hasattr(obj, 'reset'):
            obj.reset()  # Reset the instance state
            obj.release()  # Release the object back to the pool
        try:
            self.pool.put(obj, block=False)
        except queue.Full:
            print("Warning: Trying to release object to full pool")

class VAPPooledObject:
    """Wrapper for VAP instances that handles pool management"""
    
    def __init__(self, vap_wrapper: VAPWrapper, pool: VAPObjectPool):
        self.vap_wrapper = vap_wrapper
        self.pool = pool
        self.in_use = False
    
    # def acquire(self):
    #     """Mark this object as in use"""
    #     self.in_use = True
    #     return self
    
    def release(self):
        """Release this object back to the pool"""
        if self.in_use:
            self.in_use = False
            self.pool.release(self)
    
    def reset(self):
        """Reset the VAP wrapper state for reuse"""
        # Add any state reset logic here if needed
        self.vap_wrapper.reset()
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped VAP instance"""
        return getattr(self.vap_wrapper, name)