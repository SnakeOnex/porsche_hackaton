class PID:
    def __init__(self):
        self.kp = 0.3
        self.ki = 0.01
        self.kd = 0.0
        self.max = 1.0
        self.min = -1.0
        self.i = 0.
        self.e_prev = 0.

    def reset(self):
        self.i = 0.
        self.e_prev = 0.

    def set_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def step(self, setpoint, process_variable, dt):
        """
        PID controller step

        Args:
          setpoint - desired value
          process_variable - current value
          dt - time step

        Returns:
            output - PID output 
        """

        # Regulation step
        error = setpoint - process_variable
        self.i += error * dt
        output = self.kp * error + self.ki * self.i + self.kd * (error - self.e_prev) / dt
        self.e_prev = error

        # Anti-windup (conditional integration) and saturation
        if output > self.max:
            output = self.max
            self.i -= error * dt
        elif output < self.min:
            output = self.min
            self.i -= error * dt

        return output
