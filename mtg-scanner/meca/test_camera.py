import picamera

camera = picamera.PiCamera()
camera.capture('/home/pi/Desktop/scanner/images/capture.jpg', quality=100)
print("END")
