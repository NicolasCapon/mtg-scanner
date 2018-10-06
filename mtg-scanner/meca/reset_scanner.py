from __future__ import print_function # use python 3 syntax but make it compatible with python 2
from __future__ import division       #                           ''

import time     # import the time library for the sleep function
import brickpi3 # import the BrickPi3 drivers

BP = brickpi3.BrickPi3() # Create an instance of the BrickPi3 class. BP will be the BrickPi3 object.
BP.set_motor_power(BP.PORT_C, -50)
time.sleep(0.5)
BP.set_motor_power(BP.PORT_A, 50)
time.sleep(0.2)
BP.set_motor_power(BP.PORT_C, 0)
time.sleep(1)
BP.set_motor_power(BP.PORT_A, 0)

BP.reset_all()