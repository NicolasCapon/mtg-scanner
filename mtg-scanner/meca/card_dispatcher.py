import brickpi3
import time

BP = brickpi3.BrickPi3()
motor = BP.PORT_A

def go_left(angle):
    BP.offset_motor_encoder(motor, BP.get_motor_encoder(motor))
    BP.set_motor_limits(motor, 50, 400)
    BP.set_motor_position(motor, angle)

def go_right(angle):
    BP.offset_motor_encoder(motor, BP.get_motor_encoder(motor))
    BP.set_motor_limits(motor, 50, 400)
    BP.set_motor_position(motor, - angle)
    
try:
    while True:
        go_left(365)
        time.sleep(1.5)
        go_right(365)
        time.sleep(1.5)
        
except KeyboardInterrupt:
    BP.reset_all()
    pass
