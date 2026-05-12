
from pyftdi.gpio import GpioController
from pyftdi.spi import SpiController

import time 
gpio = GpioController()
gpio.open_from_url('ftdi://ftdi:232h/1')

gpio.set_direction(0xFF, 0xFF)  # all outputs
# gpio.write(0b0000_0001)
# time.sleep(0.5)
# gpio.write(0b0000_0010)
# time.sleep(0.5)
# gpio.write(0b0000_0100)
# time.sleep(0.5)
# gpio.write(0b0000_1000)
# time.sleep(0.5)
# gpio.write(0b0001_0000)
# time.sleep(0.5)
# gpio.write(0b0010_0000)
# time.sleep(0.5)
# gpio.write(0b0100_0000)
# time.sleep(0.5)
gpio.write(0b1000_0000)
time.sleep(1)

gpio.close()



# Create SPI controller
spi = SpiController()

# Configure the FTDI device.
# Replace this URL with the one shown by:
#   python3 -m pyftdi.urls
spi.configure('ftdi://ftdi:232h/1')

# Get SPI port.
#
# cs=0 means use chip-select 0.
# freq is SPI clock frequency in Hz.
# mode defines SPI mode:
#   mode=0 -> CPOL=0, CPHA=0
#   mode=1 -> CPOL=0, CPHA=1
#   mode=2 -> CPOL=1, CPHA=0
#   mode=3 -> CPOL=1, CPHA=1
slave = spi.get_port(cs=0, freq=1E6, mode=0)

# Write data only

i = 0
while True:
    k = bytes([i, i,i , i])
    i = i + 1
    if i == 256: 
        i = 0
    slave.write(k)
    print("SPI data sent:", i, k.hex())
    time.sleep(0.05)

print("SPI data sent:")

# Close interface
spi.terminate()

