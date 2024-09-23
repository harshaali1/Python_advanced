import serial
import matplotlib.pyplot as plt
from drawnow import *

# Configure the serial port
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with your port and 9600 with your baud rate

# Initialize data lists
sensor_data = []

# Create a function to plot the data
def plot_sensor_data():
    plt.ion()  # Turn on interactive mode for real-time updates
    plt.figure()
    plt.title('Live Sensor Data')
    plt.xlabel('Time')
    plt.ylabel('Sensor Reading')

    while True:
        if ser.in_waiting > 0:
            # Read and decode data from the serial port
            line = ser.readline().decode('utf-8').rstrip()

            try:
                # Convert the data to a float
                sensor_reading = float(line)

                # Append the reading to the data list
                sensor_data.append(sensor_reading)

                # Limit the data points displayed for performance
                if len(sensor_data) > 50:
                    sensor_data.pop(0)

                # Update the plot with the new data
                plt.plot(sensor_data)
                plt.pause(0.01)  # Small pause for real-time display
                drawnow()  # Update the plot in real-time
            except ValueError:
                print(f"Invalid sensor reading: {line}")

# Start the data plotting in a separate thread
import threading
plot_thread = threading.Thread(target=plot_sensor_data)
plot_thread.daemon = True  # Allow the program to exit even if the plot thread is running
plot_thread.start()

# Keep the main thread running
while True:
    pass
