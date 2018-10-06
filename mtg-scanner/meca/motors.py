#!/usr/bin/env python
#
# https://www.dexterindustries.com/BrickPi/
# https://github.com/DexterInd/BrickPi3
#
# Copyright (c) 2016 Dexter Industries
# Released under the MIT license (http://choosealicense.com/licenses/mit/).
# For more information, see https://github.com/DexterInd/BrickPi3/blob/master/LICENSE.md
#
# This code is an example for running a motor a target speed (specified in Degrees Per Second) set by the encoder of another motor.
# 
# Hardware: Connect EV3 or NXT motors to the BrickPi3 motor ports A and D. Make sure that the BrickPi3 is running on a 9v power supply.
#
# Results:  When you run this program, motor A speed will be controlled by the position of motor D. Manually rotate motor D, and motor A's speed will change.

from __future__ import print_function # use python 3 syntax but make it compatible with python 2
from __future__ import division       #                           ''

import time     # import the time library for the sleep function
import brickpi3 # import the BrickPi3 drivers
import picamera

BP = brickpi3.BrickPi3() # Create an instance of the BrickPi3 class. BP will be the BrickPi3 object.
main_motor = BP.PORT_B
sec_motor = BP.PORT_C
drop_motor = BP.PORT_A
BP.set_motor_limits(BP.PORT_C, 50)

def drop_cards():
    try:
        while True:
            BP.set_motor_power(main_motor, -50)
            time.sleep(0.3)
            BP.set_motor_power(sec_motor, 50)
            time.sleep(0.5)
            BP.set_motor_power(main_motor, 0)
            time.sleep(1)
            BP.set_motor_power(sec_motor, 0)
            input("Press Enter to drop a card")
            capture()
            drop()

    except KeyboardInterrupt:
        BP.reset_all()
        pass

def drop():
    # Make a 360 turn
    BP.offset_motor_encoder(drop_motor, BP.get_motor_encoder(drop_motor))
    BP.set_motor_limits(drop_motor, 50, 400)
    BP.set_motor_position(drop_motor, -365)

def capture():
    """Take picture"""
    camera = picamera.PiCamera()
    camera.capture('/home/pi/Desktop/scanner/images/capture.jpg', quality=100)


drop_cards()
BP.reset_all()
