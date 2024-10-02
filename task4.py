import cv2
import numpy as np
import argparse

# Parse command-line arguments for debug mode
parser = argparse.ArgumentParser(description='Animated Digital Curtain with Debug Mode')
parser.add_argument('--debug', action='store_true', help='Enable debug mode to visualize processing steps')
args = parser.parse_args()
debug_mode = args.debug

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

# Create background subtractor for motion detection
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=200)

def blend_transparent(background, overlay):
    """
    Overlay 'overlay' image on 'background' image with transparency.

    Parameters:
    - background: The background image.
    - overlay: The image to overlay with alpha channel.

    Returns:
    - Blended image.
    """
    # Split the overlay image into color channels and alpha channel
    overlay_img = overlay[..., :3]  # Color channels
    overlay_mask = overlay[..., 3:]  # Alpha channel

    # Normalize the alpha mask to keep intensity between 0 and 1
    mask = overlay_mask / 255.0

    # Get the region of interest from the background where the overlay will be placed
    background_roi = background[0:overlay.shape[0], 0:overlay.shape[1]]

    # Blend the images using the alpha mask
    blended = background_roi * (1 - mask) + overlay_img * mask

    # Return the blended image
    return blended.astype(np.uint8)

# Load curtain image with alpha channel (transparency)
curtain = cv2.imread('curtain.png', cv2.IMREAD_UNCHANGED)
if curtain is None or curtain.size == 0:
    print('Error: Curtain image not found or is empty.')
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Set fixed size for the curtain
curtain_width = 400
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
curtain_height = frame_height
curtain_resized_fixed = cv2.resize(curtain, (curtain_width, curtain_height))

# Initialize curtain position and animation parameters
curtain_position = 0
target_position = 0
animation_speed = 0.1  # Controls the speed of the curtain movement

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_hist_eq = cv2.equalizeHist(frame_gray)

    # Reduce noise by applying a Gaussian Blur before processing the frame
    blurred_frame = cv2.GaussianBlur(frame_hist_eq, (5, 5), 0)

    # Apply background subtraction to get the foreground mask
    fgmask = background_subtractor.apply(blurred_frame)

    # Tresholding for foreground segmentation: manual thresholding to improve motion detection
    _, fgmask_thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to reduce noise in the foreground mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask_clean = cv2.morphologyEx(fgmask_thresh, cv2.MORPH_OPEN, kernel)

    # Find contours in the clean foreground mask
    contours, _ = cv2.findContours(fgmask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour, assuming it's the main moving object
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Update target position based on detected motion
        target_position = x

        if debug_mode:
            # Draw bounding rectangle around the largest contour
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate the center of the contour
    contour_center = x + w // 2

    # Calculate the desired curtain position (centered over the contour)
    target_position = contour_center - curtain_width // 2

    # Smoothly animate curtain position towards the target position
    curtain_position += (target_position - curtain_position) * animation_speed

    # Calculate curtain placement
    x_start = int(curtain_position)
    y_start = 0

    # Ensure the curtain does not go beyond the frame boundaries
    x_start = max(0, min(x_start, frame.shape[1] - curtain_width))
    x_end = min(x_start + curtain_width, frame.shape[1])
    curtain_width_actual = x_end - x_start

    # Adjust the curtain image if necessary
    curtain_resized = curtain_resized_fixed[:, :curtain_width_actual]

    # Blend the curtain onto the frame
    try:
        frame[y_start:y_start+curtain_resized.shape[0], x_start:x_end] = blend_transparent(
            frame[y_start:y_start+curtain_resized.shape[0], x_start:x_end], curtain_resized
        )
    except ValueError:
        # Skip blending if dimensions do not match
        pass

    # Apply mirror effect to the frame
    mirrored_frame = cv2.flip(frame, 1)

    # Display the resulting frame
    cv2.imshow('Animated Digital Curtain', mirrored_frame)

    if debug_mode:
        # Display the foreground mask
        cv2.imshow('Foreground Mask', fgmask)

        # Display the cleaned foreground mask after morphological operations
        cv2.imshow('Cleaned Foreground Mask', fgmask_clean)

        # Create a copy of the frame to draw contours
        contours_frame = frame.copy()
        cv2.drawContours(contours_frame, contours, -1, (0, 0, 255), 2)
        cv2.imshow('Contours', contours_frame)

    # Keyboard controls
    key = cv2.waitKey(20)
    if key == 27:  # ESC key to exit
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
