
import time
import os
import pandas as pd
from collections import OrderedDict


class Timer(object):

    def __init__(self):
        self.start_time = None
        self.total = 0
        self.count_ = 0
       
    
    def start(self):
        '''
        Start timer
        '''
        self.start_time = time.perf_counter()
        
    def stop(self):
        '''
        Stop timer and get time Delta
        '''
        # Add time Delta and increment counter
        assert self.start_time is not None
        self.total += time.perf_counter() - self.start_time
        self.count_ += 1
        
        # Make sure the next timer is started before it is stopped
        self.start_time = None

    def average(self):
        '''
        Get average processing time
        '''
        
        assert self.count_ > 0
        return self.total/float(self.count_)
        
    def __str__(self):
        
        summary = []
        summary.append('Timer summary')
        summary.append('\tstart_time:\t{}'.format(self.start_time))
        summary.append('\ttotal:\t{}'.format(self.total))
        summary.append('\tcount_:\t{}'.format(self.count_))
        summary.append('\taverage:\t{}'.format(self.average()))
        return "\n".join(summary)

class Timers(object):
    
    
    def __init__(self, path=None):
        self.path_ = path
        self.timers = OrderedDict()

        
        
    def start(self, name):
        '''
        Start timer specified by "name"
        '''            
        
        # Create timer if does not exist
        if name not in self.timers:
            self.timers[name] = Timer()
            
        # Start timer            
        self.timers[name].start()
        

    def stop(self, name):
        '''
        Stop timer specified by "name"
        '''            
        
        # Check for timers existence
        assert name in self.timers
            
        # Stop timer            
        self.timers[name].stop()

    def save(self, path=None):
        
        if path is None:
            path = self.path_
        
        assert path is not None
        
        # Get average execution times
        averages = [(k, v.average()) for k, v in self.timers.items()]
        
        # Save execution times
        df = pd.DataFrame(averages, columns=['name', 'time (s)'])
        fn = os.path.join(path, 'timing_summary.csv')
        df.to_csv(fn, index=False)

        return df 
        
    def reset(self):
        self.timers = OrderedDict()
        

    def __str__(self):
        summary = []        
        for k, v in self.timers.items():
            summary.append('')
            summary.append(k)
            summary.append(str(v))
        return "\n".join(summary)            
        