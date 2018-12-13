
# coding: utf-8

# In[5]:


import math
import matplotlib.pyplot as plt
from random import randint
import time
import numpy as np
from scipy.interpolate import spline
import csv

class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        error = self.SetPoint - feedback_value

        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time
        
class lump:
    """Lumped capacitance method, takes 5 variables - resistance, capacitance, ambient temperature,
    starting temperature, and a time step. 
    """
    def __init__(self, R, C, t_inf, _tinitial, dt):
        self.R = R
        self.C = C
        self.t_inf = t_inf
        self._tinitial = _tinitial
        self.dt = dt
        self.current_t = self._tinitial
    
    def updateT(self, q):
        """Returns a new temperature based on a set q input. Resets the internal body temperature to
        new calculated temperature.
        """
        t_dif_old = self.current_t-self.t_inf
        t_new = ((t_dif_old)*math.exp((-1*self.dt)/(self.C)))+q*self.R+self.t_inf
        self.current_t = t_new
        return t_new

if __name__ == "__main__":
    # Set PID controller with variables
    pid = PID(1.2, 0.5, 0.001)
    # What the PID controller is trying to get to
    pid.SetPoint=37
    # Sample time rate is number of times per second Controller runs
    pid.setSampleTime(0.01)
    feedback = 20
    
    test_model = lump(1, 1, 10, 33, 0.1)
    
    # Lists for data storage
    feedback_list = []
    time_list = []
    setpoint_list = []
    
    # Currently loop runs 1000 times with controller updating every time.
    # After the 500th run, the ambient temperature switches. Possible improvements
    # to this part of the code include writing a method which can change the 
    # ambient temperature after the internal temperature gets to within a certain
    # percentage of internal steady state temperature. Psuedo code envisioned is as
    # follows- a while loop nested inside of a for loop, the internal while loop will run until 
    # test_model.current_t <= 0.001*pid.SetPoint. Outside of this for loop will run 5-7 times
    # and each time it runs, test_model.t_inf will update to a random temperature in the
    # range of 30-70 
    for i in range(1, 1000):
        pid.update(feedback)
        output = pid.output
        feedback = test_model.updateT(output)
        time.sleep(0.02)
        if i >= 500:
            test_model.t_inf = 70
        feedback_list.append(feedback)
        setpoint_list.append(output)
        time_list.append(i)

    time_sm = np.array(time_list)
    time_smooth = np.linspace(time_sm.min(), time_sm.max(), 300)
    feedback_smooth = spline(time_list, feedback_list, time_smooth)
    feedback_smooth1 = spline(time_list, setpoint_list, time_smooth)
    
    plt.plot(time_sm, feedback_list, 'g')
    plt.plot(time_sm, setpoint_list)
    
    print(feedback_list[398])
    print(feedback_list[198])
    print(setpoint_list[298])
    print(setpoint_list[30])
    
    plt.show()
    
    with open('testdata_2.csv', mode='w') as csv_file:
        data_writer = csv.writer(csv_file, delimiter=',')
        data_writer.writerow(['t1','t2','q'])
        data_writer.writerow([0, feedback_list[0], setpoint_list[0]])
        for x in range(998):
            data_writer.writerow([feedback_list[x], feedback_list[x+1], setpoint_list[x+1]])
        
            


# In[ ]:


import time
"""Different PID controller than the above one. Does not work as well."""
class PID:
    """ Simple PID control.

        This class implements a simplistic PID control algorithm. When first
        instantiated all the gain variables are set to zero, so calling
        the method GenOut will just return zero.
    """
    def __init__(self):
        # initialze gains
        self.Kp = 0
        self.Kd = 0
        self.Ki = 0

        self.Initialize()

    def SetKp(self, invar):
        """ Set proportional gain. """
        self.Kp = invar

    def SetKi(self, invar):
        """ Set integral gain. """
        self.Ki = invar

    def SetKd(self, invar):
        """ Set derivative gain. """
        self.Kd = invar

    def SetPrevErr(self, preverr):
        """ Set previous error value. """
        self.prev_err = preverr

    def Initialize(self):
        # initialize delta t variables
        self.currtm = time.time()
        self.prevtm = self.currtm

        self.prev_err = 0

        # term result variables
        self.Cp = 0
        self.Ci = 0
        self.Cd = 0


    def GenOut(self, error):
        """ Performs a PID computation and returns a control value based on
            the elapsed time (dt) and the error signal from a summing junction
            (the error parameter).
        """
        self.currtm = time.time()               # get t
        dt = self.currtm - self.prevtm          # get delta t
        de = error - self.prev_err              # get delta error

        self.Cp = self.Kp * error               # proportional term
        self.Ci += error * dt                   # integral term

        self.Cd = 0
        if dt > 0:                              # no div by zero
            self.Cd = de/dt                     # derivative term

        self.prevtm = self.currtm               # save t for next pass
        self.prev_err = error                   # save t-1 error

        # sum the terms and return the result
        return self.Cp + (self.Ki * self.Ci) + (self.Kd * self.Cd)

class lump:
    def __init__(self, R, C, t_inf, _tinitial, dt):
        self.R = R
        self.C = C
        self.t_inf = t_inf
        self._tinitial = _tinitial
        self.dt = dt
        self.current_t = self._tinitial
    
    def updateT(self, q):
        t_dif_old = self.current_t-self.t_inf
        t_new = ((t_dif_old)*math.exp((-1*self.dt)/(self.C)))+q*self.R+self.t_inf
        self.current_t = t_new
        return t_new

