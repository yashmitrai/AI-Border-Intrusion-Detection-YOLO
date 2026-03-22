import time

class ZoneStateManager:
    def __init__(self):
        self.current_state = "SAFE"
        self.last_state_change_time = 0
        self.consecutive_WARNING = 0
        self.consecutive_INTRUSION = 0
        
        self.last_intrusion_seen = 0
        self.last_warning_seen = 0
        
        self.STATE_PRIORITY = {"SAFE": 0, "WARNING": 1, "INTRUSION": 2}
        
    def update(self, raw_state):
        now = time.time()
        
        # Track consecutive frames for escalation
        if raw_state == "INTRUSION":
            self.consecutive_INTRUSION += 1
            self.consecutive_WARNING = 0
            self.last_intrusion_seen = now
        elif raw_state == "WARNING":
            self.consecutive_WARNING += 1
            self.consecutive_INTRUSION = 0
            self.last_warning_seen = now
        else: # SAFE
            self.consecutive_INTRUSION = 0
            self.consecutive_WARNING = 0

        current_priority = self.STATE_PRIORITY[self.current_state]

        # Prevent state toggles within 2 seconds of last actual state change (Debounce)
        if (now - self.last_state_change_time) < 2.0:
            return self.current_state

        # Escalation Logic -> Requires 3 consecutive frames
        if raw_state == "INTRUSION" and current_priority < 2:
            if self.consecutive_INTRUSION >= 3:
                self.current_state = "INTRUSION"
                self.last_state_change_time = now
        elif raw_state == "WARNING" and current_priority < 1:
            if self.consecutive_WARNING >= 3:
                self.current_state = "WARNING"
                self.last_state_change_time = now
        
        # Downgrade Logic -> Requires 3 seconds of continuous absence of higher priority state
        if self.current_state == "INTRUSION":
            if (now - self.last_intrusion_seen) > 3.0:
                # Downgrade to WARNING if warning was recently seen, otherwise SAFE
                if (now - self.last_warning_seen) <= 3.0:
                    self.current_state = "WARNING"
                else:
                    self.current_state = "SAFE"
                self.last_state_change_time = now
                
        elif self.current_state == "WARNING":
            if (now - self.last_warning_seen) > 3.0 and (now - self.last_intrusion_seen) > 3.0:
                self.current_state = "SAFE"
                self.last_state_change_time = now

        return self.current_state
