#!/usr/bin/env python3
import serial
import math
from time import sleep
# from .PG9038S import PG9038S as bt
import subprocess as sp
import time

print("    4 Wheel Drive Remote Control for Serial-Curtis Bridge v1.3 and Generic Bluetooth Controller")
print("    Four wheel drive electronic differential with ackermann steering via linear actuator and ancilliary lift actuator")
print("    Usage: Left or Right Trigger = Toggle Enable")
print("    Usage: Left Joystick for forward and reverse motion")
print("    Usage: Right Joystick for steering left and right")
print("    Usage: DPad up/down to raise/lower tool")
print("    Usage: estop enable = either joystick buttons, cancel estop = both bumper buttons")
print("     - Rob Lloyd. 02/2021")

## Describe the critical dimensions of the vehicle 4WD
vehicleWidth = 1.5
vehicleLength = 2.0

# change to unique MAC address of bluetooth controller
controllerMAC = "DD:44:63:38:84:07" 

# create an object for the bluetooth control
# controller = bt("/dev/input/event1")

# create an object for the serial port controlling the curtis units
try:
    curtisData = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
except:
    print("Curtis-Serial Bridge Failed to connect")
    pass

# create an object for the serial port controlling the curtis units
try:
    actData = serial.Serial("/dev/ttyUSB1", 115200, timeout=1)
except:
    print("Actuator Controller Failed to connect")
    pass

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

class SerialComm(object):

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

        self._debug('Opening serial connection')
        self.sock = serial.Serial(port, baud_rate, timeout=1)
        self._debug('Serial connection opened successfully')

        time.sleep(2)  # Wait for the Arduino bootloader 



def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
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




def generateCurtisMessage(estopState, enable, v1, v2, v3, v4):
    """
    Accepts an input of two bools for estop and enable. 
    Then two velocities for right and left wheels between -100 and 100
    """
    # Empty list to fill with our message
    messageToSend = []
    # # # Check the directions of the motors, False (0) = (key switch) forward, True (1) = reverse
    # # # Velocities are scaled from -100 to +100 with 0 (middle of joystick) = no movement
    # # # If vel >= 0 = forward, if vel < 0 backward
    vels = [v1, v2, v3, v4]
    dirs = []
    for i in vels:
        if vels[i] >= 0:
            dirs[i] = False
        elif vels[i] < 0:
            dirs[i] = True


    # Check to see if we're allowed to move. estop and enable
    if estopState or not enable:
        for i in vels:
            vels[i] = 0

    # # Build the message. converting everything into positive integers
    # # Message is 10 bits [estopState, enable, motor 0 direction, motor 0 velocity, motor 1 direction, motor 1 velocity, motor 2 direction, motor 2 velocity, motor 3 direction, motor 3 velocity]
    # # motor numbering:
    # #  Front    
    # # 0  1
    # # 2  3
    # # Back (key end)

    messageToSend.append(int(estopState))
    messageToSend.append(int(enable))
    messageToSend.append(int(dirs[0]))
    messageToSend.append(abs(int(vels[0])))
    messageToSend.append(int(dirs[1]))
    messageToSend.append(abs(int(vels[1])))
    messageToSend.append(int(dirs[2]))
    messageToSend.append(abs(int(vels[2])))
    messageToSend.append(int(dirs[3]))
    messageToSend.append(abs(int(vels[3])))
    
    print("Sending: %s" % str(messageToSend))
    return messageToSend

def generateActMessage(enable, angle, height):
    """
    Accepts an input of two ints between -100 and 100
    """
    # Empty list to fill with our message
    messageToSend = []

    messageToSend.append(int(enable))
    messageToSend.append(int(angle))
    messageToSend.append(int(height))
    
    print("Sending: %s" % str(messageToSend))
    return messageToSend

def send(message_in, conn):
    """
    Function to send a message_in made of ints, convert them to bytes and then send them over a serial port
    message length, 10 bytes.
    """
    if conn == 0:
        messageLength = 10
        message = []
        for i in range(0, messageLength):
            message.append(message_in[i].to_bytes(1, 'little'))
        for i in range(0, messageLength):
            curtisData.write(message[i])
    elif conn == 1:
        messageLength = 3
        message = []
        for i in range(0, messageLength):
            message.append(message_in[i].to_bytes(1, 'little'))
        for i in range(0, messageLength):
            actData.write(message[i])
    #print(message)

def receive():
    """
    Function to read whatever is presented to the serial port and print it to the console.
    Note: For future use: Currently not used in this code.
    """
    messageLength = len(message)
    last_message = []
    try:
        while arduinoData.in_waiting > 0:
            for i in range(0, messageLength):
                last_message.append(int.from_bytes(arduinoData.read(), "little"))
        #print("GOT: ", last_message)
        return last_message
    except:
        print("Failed to receive serial message")
        pass

def isEnabled():
    """ 
    Function to handle enable and estop states. it was geeting annoying to look at.
    """
    # to reset after estop left and right bumper buttons - press together to cancel estop
    if newStates["trigger_l_1"] == 1 and newStates["trigger_r_1"] == 1:
        estopState = False

    # left and right joystick buttons trigger estop
    if newStates["button_left_xy"] == 1 or newStates["button_right_xy"] == 1:
        estopState = True #this shouldnt reset. but it does
    
    if estopState == True:
        enable = False #ok

    # dead mans switch left or right trigger button
    if newStates["trigger_l_2"] == 1 or newStates["trigger_r_2"] == 1:
        if estopState == False:
            enable = True
    else:
        enable = False

    return enable

def calculateVelocities(velocity, angle):
    # Appl Sci 2017, 7, 74
    if angle <= 0: #turn Left
        R = L/math.tan(angle)
        v1 = velocity*(1-(vehicleWidth/R))
        v2 = velocity*(1+(vehicleWidth/R))
        v3 = velocity*((R-(vehicleWidth/2)/R))
        v4 = velocity*((R+(vehicleWidth/2)/R))
    elif angle >= 0: #turn Right
        R = L/math.tan(angle)
        v1 = velocity*(1+(vehicleWidth/R))
        v2 = velocity*(1-(vehicleWidth/R))
        v3 = velocity*((R+(vehicleWidth/2)/R))
        v4 = velocity*((R-(vehicleWidth/2)/R))

    return v1, v2, v3, v4

# Main Loop
while True:
    stdoutdata = sp.getoutput("hcitool con") # hcitool check status of bluetooth devices

    # check bluetooth controller is connected if not then estop
    if controllerMAC not in stdoutdata.split():
        print("Bluetooth device is not connected")
        enable = False
        estopState = True
    else:
        enable = True

    # Check to see if there is new input from the controller
    try:
        newStates = controller.readInputs()
    except IOError:
        pass

    if newStates["dpad_y"] > 129:
        toolPos += 1
    elif newStates["dpad_y"] < 127:
        toolPos -= 1
    # Rescal the tool position. 100 is full up, 0 is full down. #'###CHECK THIS    
    commandTool = rescale(toolPos, 255, 0, 100, 0)
    
    # Check the enable state via the function
    if isEnabled: 
        # Calculate the final inputs rescaling the absolute value to between -100 and 100
        commandVel = rescale(newStates["left_y"], 255,0,-100,100)
        commandAngle = rescale(newStates["right_x"], 255,0,-100,100)
        v1, v2, v3, v4 = calculateVelocities(commandVel, commandAngle)

    else:
        commandVel = 0
        commandAngle = rescale(newStates["right_x"], 255,0,-100,100)
        v1, v2, v3, v4 = calculateVelocities(commandAngle, 0)

    # Build a new message with the correct sequence for the curtis Arduino
    newCurtisMessage = generateCurtisMessage(estopState, enable, v1, v2, v3, v4)
    # Build new message for the actuators
    newActMessage = generateActMessage(enable, commandTool, commandAngle)
    # Send the new message to the actuators and curtis arduinos
    send(newActMessage, 1)
    send(newCurtisMessage, 0)
    # So that we don't keep spamming the Arduino....
    sleep(0.1)