# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 23:18:06 2023

@author: nrb50
"""

import cv2
import numpy as np
import keyboard

# Initialize variables
data = np.zeros((400, 400), dtype=np.uint8)
current_key = None

# Function to update data based on keypress
def update_data(key_event):
    global data, current_key

    if key_event.name == 'a':
        #data = np.zeros((400, 400), dtype=np.uint8)
        data = np.zeros((400, 400), dtype=np.uint8)
        current_key = 'a'
    elif key_event.name == 's':
        data = np.full((400, 400), 255, dtype=np.uint8)
        current_key = 's'

# Hook key events
keyboard.on_press(update_data)

# Initialize video window
window_name = 'Dynamic Data Display'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (400, 400))

# Loop to display data
while True:
    # Convert data to a BGR image
    #image = cv2.merge([data, data, data])
    image = cv2.merge([data,data,data])

    # Display the image
    cv2.imshow(window_name, image)

    # Write the frame to the video file
    video_writer.write(image)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video writer and close windows
video_writer.release()
cv2.destroyAllWindows()