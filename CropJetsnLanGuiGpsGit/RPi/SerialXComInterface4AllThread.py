#!/usr/bin/env python3
import serial
import math
from time import sleep
# from .PG9038S import PG9038S as bt
import subprocess as sp
import time
import sys
import glob

import threading

STRGLO="" #Read data
BOOL=True  #Read flag bit


# So the direction in general can be reversed
direction = False

# Initialise  values for enable and estop
estopState = False
enable = False
left_y = 128
right_x = 128
toolPos = 128

# curtisMessage = []  # Seems to be necessary to have a placeholder for the message here
# actMessage = []
# last_message = []
# https://www.programcreek.com/python/example/1568/serial.Serial

## Functions -----------------------------------------------------------------------
DEFAULT_SECTOR_SIZE = 1024
ENCODING = 'iso-8859-1'

DEBUG_NORMAL = 1
DEBUG_VERBOSE = 2
def logMessage(text):
    print (text)
def logOk(text):
    print (text)
def logError(text):
    print (str(text))
def logDebug(text, type):
    if type == DEBUG_NORMAL:
        print (text)
    else:  # DEBUG_VERBOSE
        print (text)

class SerialCommObj(object):
    # https://programmer.group/python-s-serial-communication.html
    def __init__(self, port, baud_rate, debug='off', sector_size=DEFAULT_SECTOR_SIZE, page_size=1024):
        self.sector_size = sector_size
        self.page_size = page_size
        self.pages_per_sector = self.sector_size // self.page_size

        if debug == 'normal':
            self.debug = DEBUG_NORMAL
        elif debug == 'verbose':
            self.debug = DEBUG_VERBOSE
        else:  # off
            self.debug = 0
        
        self.serialport_vec=self.serial_ports()
        self.serialport_num = len(self.serialport_vec)
        
        print ('Number of serail port available: ', self.serialport_num)
        if self.serialport_num==0:
            print ('None of serail port available: ')
            return     
        for i in range(self.serialport_num):
            print ('Serail port available: ', self.serialport_vec[i])        
        


    def _debug(self, message, level=DEBUG_NORMAL):
        if self.debug >= level:
            logDebug(message, level)

    def startSerialCom(self,port_str, baudrate):

        """# port = str(port).encode('utf-16')
        # port_str= ''.join(pt)
        
        self._debug('Opening serial connection')
        self.sock = serial.Serial(port, baud_rate, timeout=1)     

        self.sock = serial.Serial(
            port=port_str,baudrate=9600)   #port='/dev/ttyACM0',
            # port='/dev/ttyUSB0',baudrate=9600)   #port='/dev/ttyACM0',"""
        try:
            # configure the serial connections (the parameters differs on the device you are connecting to)

            """
                self.sock  = serial.Serial(
                port='/dev/ttyUSB0',
                baudrate=9600,
                parity=serial.PARITY_ODD,
                stopbits=serial.STOPBITS_TWO,
                bytesize=serial.SEVENBITS)"""

            self.sock  = serial.Serial(port=port_str, baudrate=baudrate, timeout=1)   
            # actData = serial.Serial("/dev/ttyUSB1", 115200, timeout=1)
            if (self.sock ):
                print("Serial port opened success!!")
                threading.Thread(target=self.ReadData, args=(self.sock,)).start()
                time.sleep(0.5)  # Wait for the Reciever Device bootloader 
                return self.sock 

        except:
            # create an object for the serial port controlling the curtis units
            print("Curtis-Serial Bridge Failed to connect")
            pass
    
    def DPCWritePort(self,ser,dir,pwm1,pwm2):  ## dir: 0(forward), 1(right), 2(left), 3(backward)
        # https://haeyeonlab.wordpress.com/2019/12/03/2019-12-03-serial-communication-between-jetson-tx2-and-arduino-python/
        
        # read and write on serial
        # Read from and write to a serial port
        # https://web.dev/serial/
        try:
            # print("PC Sending via Serial Communication :")     
            output=str(dir)+","+str(pwm1)+","+str(pwm2)
            print('Sending out: ', output)             
            # ser.write(str.encode(output))
            result = ser.write(output.encode('utf-8'))      
                  
        except:
            print("Tx2 Failed to be connected")
            result = None
            pass
        return None

    def ReadData(self,ser, chunk_size=8):

        global STRGLO,BOOL        
        
        """Read all characters on the serial port and return them."""
        if not ser.timeout:
            raise TypeError('Port needs to have a timeout set!')
        # Loop receives data, which is a dead loop and can be implemented by threads
        read_buffer = b''
        while BOOL:
            if ser.in_waiting:
                STRGLO = ser.read(ser.in_waiting).decode('utf-8')
                print('Thread Reading:',STRGLO)

        read_buffer = STRGLO
            
        # https://stackoverflow.com/questions/19161768/pyserial-inwaiting-returns-incorrect-number-of-bytes
        """
        You cannot randomly partition the bytes you've received and then ask UTF-8 to decode it. 
        UTF-8 is a multibyte encoding, meaning you can have anywhere from 1 to 6 bytes 
        to represent one character. If you chop that in half, and ask Python to decode it, 
        it will throw you the unexpected end of data error.
        """
        """
        read_buffer = b''
        while True:
            # Read in chunks. Each chunk will wait as long as specified by
            # timeout. Increase chunk_size to fail quicker
            byte_chunk = ser.read(size = 8)
            read_buffer += byte_chunk
            if not len(byte_chunk) == chunk_size:
                break
        """
        return read_buffer

    def serial_ports(self):
        """ Lists serial port names

            :raises EnvironmentError:
                On unsupported or unknown platforms
            :returns:
                A list of the serial ports available on the system

            : for super user permission:
                https://superuser.com/questions/431843/configure-permissions-for-dev-ttyusb0

                Temporary manual change:

                To do a one-time change, just use chmod and/or chown as you normally would.

                    # chown YourUsername /dev/ttyUSB0
                or:  # chmod a+rw /dev/ttyUSB0
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin') or sys.platform.startswith('ubuntu'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/ttyUSB*') # ubuntu is /dev/ttyUSB0
        elif sys.platform.startswith('darwin'):
            # ports = glob.glob('/dev/tty.*')
            ports = glob.glob('/dev/tty.SLAB_USBtoUART*')
        else:
            raise EnvironmentError('Unsupported platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except serial.SerialException as e:
                if e.errno == 13:
                    raise e
                pass
            except OSError:
                pass
        return result

