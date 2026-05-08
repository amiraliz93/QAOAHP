
from pyftdi.gpio import GpioController
import time 
gpio = GpioController()
gpio.open_from_url('ftdi://ftdi:232h/1')

gpio.set_direction(0xFF, 0xFF)  # all outputs
gpio.write(0x00)
time.sleep(1)
gpio.write(0x00)
time.sleep(1)
gpio.write(0xff)
time.sleep(1)

gpio.close()
