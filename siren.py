import cv2
import pygame

# Set the video source (0 for built-in webcam)
video_source = 0

# Initialize the video capture object
video_capture = cv2.VideoCapture(video_source)

# Initialize the previous frame
previous_frame = None

# Initialize the pygame mixer
pygame.mixer.init()

# Load the custom sound file
sound_file = "C:/Users/ambri/OneDrive/Desktop/python/police-operation-siren-144229.wav"
# Loop until interrupted
while True:
    # Read the current frame from the video source
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Set the previous frame as the first frame
    if previous_frame is None:
        previous_frame = gray
        continue

    # Compute the absolute difference between the current and previous frame
    frame_delta = cv2.absdiff(previous_frame, gray)

    # Apply a threshold to the frame delta
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < 500:
            continue

        # Draw a bounding box around the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Play the custom sound file
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()

    # Show the video feed with motion detection
    cv2.imshow("Motion Detection", frame)

    # Set the current frame as the previous frame for the next iteration
    previous_frame = gray

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
