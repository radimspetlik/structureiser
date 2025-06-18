from time import perf_counter, time

class Stopwatch:
    def __init__(self, resolution='low', start_at_creation=True):
        self.time = time if resolution == 'low' else perf_counter
        self.started = 0.
        self.running = False
        self.last_request = 0.
        if start_at_creation:
            self.start()

    def start(self):
        if self.running: return
        self.running = True
        self.started = self.time()

    def elapsed(self):
        if not self.running: return 0.
        return self.time() - self.started

    def reset(self):
        self.running = False
        self.started = 0.
        
    def just_passed(self, target_time):
        elapsed = self.elapsed()
        if elapsed >= target_time and self.last_request < target_time:
            self.last_request = target_time
            return True
        return False
