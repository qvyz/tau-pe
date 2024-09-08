import numpy as np
import math

class CosineAnnealingScheduler:
    def __init__(self):
        # Initialize the learning rate parameters
        self._lr_max = 0.0001
        self._lr_min = 0.00005
        self._lr_max_warmup = 0.001
        self._rise_epochs = 10
        self._cycle_length = 20
        self._multiplier = 1
        self._amp_mod = True
        self._warmup_epochs = 50

    # Setter methods for each parameter

    def set_lr_max(self, lr_max):
        if lr_max <= 0:
            raise ValueError("Maximum learning rate must be positive.")
        self._lr_max = lr_max

    def set_lr_min(self, lr_min):
        if lr_min < 0:
            raise ValueError("Minimum learning rate must be non-negative.")
        self._lr_min = lr_min

    def set_lr_max_warmup(self, lr_max_warmup):
        if lr_max_warmup <= 0:
            raise ValueError("Learning rate max during warmup must be positive.")
        self._lr_max_warmup = lr_max_warmup

    def set_rise_epochs(self, rise_epochs):
        if rise_epochs <= 0:
            raise ValueError("Rise epochs must be positive.")
        self._rise_epochs = rise_epochs

    def set_cycle_length(self, cycle_length):
        if cycle_length <= 0:
            raise ValueError("Cycle length must be positive.")
        self._cycle_length = cycle_length

    def set_multiplier(self, multiplier):
        if multiplier <= 0:
            raise ValueError("Multiplier must be positive.")
        self._multiplier = multiplier

    def set_amp_mod(self, amp_mod):
        if not isinstance(amp_mod, bool):
            raise ValueError("Amplitude modulation flag must be a boolean.")
        self._amp_mod = amp_mod

    def set_warmup_epochs(self, warmup_epochs):
        if warmup_epochs < 0:
            raise ValueError("Warmup epochs must be non-negative.")
        self._warmup_epochs = warmup_epochs

    def cosine_annealing_schedule(self, epoch):
        """
        Compute the learning rate for the given epoch using a cosine annealing schedule.
        
        Parameters:
        epoch (int): The current epoch number.
        
        Returns:
        float: The learning rate for the given epoch.
        """
        delta_lr_warmup = self._lr_max_warmup - self._lr_min
        delta_lr = self._lr_max - self._lr_min
        
        if epoch < self._warmup_epochs:
            warmup_cosine_factor = np.cos(np.pi * (epoch - self._warmup_epochs / 2) / (self._warmup_epochs / 2))
            return self._lr_min + delta_lr_warmup / 2 + (delta_lr_warmup / 2 * warmup_cosine_factor)
        
        else:
            epoch -= self._warmup_epochs
            
            if self._multiplier == 1:
                current_cycle = 1
                cycle_start = 0
                cycle_duration = self._cycle_length
            else:
                logr = math.log((epoch / self._cycle_length * (self._multiplier - 1)) + 1, self._multiplier)
                current_cycle = math.floor(logr)
                cycle_start = self._cycle_length * (self._multiplier ** current_cycle - 1) / (self._multiplier - 1)
                cycle_duration = self._cycle_length * (self._multiplier ** current_cycle)
            
            cycle_epoch = (epoch - cycle_start) % cycle_duration
            
            amp_fact = current_cycle + 1 if self._amp_mod else 1
            
            if cycle_epoch <= self._rise_epochs:
                rise_cosine_factor = np.cos(np.pi * (cycle_epoch - self._rise_epochs) / self._rise_epochs)
                return self._lr_min + delta_lr / (2 * amp_fact) + (delta_lr / (2 * amp_fact) * rise_cosine_factor)
            else:
                fall_cosine_factor = np.cos(np.pi * (self._rise_epochs - cycle_epoch) / (cycle_duration - self._rise_epochs))
                return self._lr_min + delta_lr / (2 * amp_fact) + (delta_lr / (2 * amp_fact) * fall_cosine_factor)
