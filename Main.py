import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

TIP_IDS = [4, 8, 12, 16, 20]
FILTERS = ["Black & White", "Sparkle", "Negative", "Glitch", "Spotlight"]
current_filter = "None"
filter_rects = []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

selection_hand_landmarks = None

def apply_bw_filter(image):
    """Applies an intensified black and white filter."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

def apply_sparkle_filter(image):
    """Adds a star-like sparkle and shine effect."""
    sparkle_image = image.copy()
    height, width, _ = image.shape
    
    num_shines = 50 
    for _ in range(num_shines):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(2, 5) 
        
        cv2.circle(sparkle_image, (x, y), radius, (255, 255, 255), -1)
        
        num_rays = np.random.randint(4, 7) * 2
        angle_step = 360 / num_rays
        
        for i in range(num_rays):
            angle = i * angle_step
            end_x = int(x + radius * 3 * np.cos(np.radians(angle)))
            end_y = int(y + radius * 3 * np.sin(np.radians(angle)))
            
            start_x = int(x + radius * np.cos(np.radians(angle)))
            start_y = int(y + radius * np.sin(np.radians(angle)))
            
            cv2.line(sparkle_image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 1)

    return cv2.addWeighted(image, 0.7, sparkle_image, 0.3, 0)

def apply_negative_filter(image):
    """Inverts the colors of the image."""
    return cv2.bitwise_not(image)

def apply_glitch_filter(image):
    """Applies a simple horizontal line glitch effect."""
    glitch_image = image.copy()
    height, width, _ = glitch_image.shape
    
    num_glitch_lines = np.random.randint(20, 40)
    
    for _ in range(num_glitch_lines):
        y = np.random.randint(0, height)
        thickness = np.random.randint(2, 6)
        shift = np.random.randint(-50, 50)
        
        if shift > 0:
            glitch_image[y:y+thickness, shift:] = glitch_image[y:y+thickness, :-shift]
        elif shift < 0:
            glitch_image[y:y+thickness, :shift] = glitch_image[y:y+thickness, -shift:]

    return glitch_image

def apply_spotlight_filter(image, mask):
    """Makes the area between the hands visible and the rest of the screen black."""
    black_screen = np.zeros_like(image)
    return cv2.bitwise_and(image, image, mask=mask) + cv2.bitwise_and(black_screen, black_screen, mask=cv2.bitwise_not(mask))

def draw_ui(image, selection_point):
    """Draws the filter selection menu with button highlights."""
    global filter_rects
    filter_rects = []
    height, width, _ = image.shape
    menu_height = 80
    
    overlay = image.copy()
    cv2.rectangle(overlay, (0, height - menu_height), (width, height), (0, 0, 0), -1)
    alpha = 0.5
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    button_width = width // len(FILTERS)
    for i, filter_name in enumerate(FILTERS):
        x1 = i * button_width
        x2 = (i + 1) * button_width
        y1 = height - menu_height
        y2 = height
        rect = (x1, y1, x2, y2)
        filter_rects.append(rect)

        # Highlight the button if the selection finger is over it
        if x1 < selection_point[0] < x2 and y1 < selection_point[1] < y2:
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 255, 100), -1)  # Green filled
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White border
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White border

        # Draw text
        text_size = cv2.getTextSize(filter_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (button_width - text_size[0]) // 2
        text_y = y1 + (menu_height + text_size[1]) // 2
        cv2.putText(image, filter_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image

def check_selection(point):
    """Checks if the given point is within any of the filter UI rectangles."""
    for i, rect in enumerate(filter_rects):
        x1, y1, x2, y2 = rect
        if x1 < point[0] < x2 and y1 < point[1] < y2:
            return FILTERS[i]
    return "None"

def apply_selective_filter(image, landmarks1, landmarks2, filter_type):
    """
    Applies the selected filter to the area between the thumb and index finger
    of both hands, creating a precise quadrilateral window.
    """
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    points = []

    tip1_index_x = int(landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
    tip1_index_y = int(landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
    points.append([tip1_index_x, tip1_index_y])
    
    tip1_thumb_x = int(landmarks1.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
    tip1_thumb_y = int(landmarks1.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
    points.append([tip1_thumb_x, tip1_thumb_y])
    
    tip2_thumb_x = int(landmarks2.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
    tip2_thumb_y = int(landmarks2.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
    points.append([tip2_thumb_x, tip2_thumb_y])
    
    tip2_index_x = int(landmarks2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
    tip2_index_y = int(landmarks2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
    points.append([tip2_index_x, tip2_index_y])
    
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points], 255)
    
    if filter_type == "Spotlight":
        return apply_spotlight_filter(image, mask)

    filtered_image = image.copy()
    if filter_type == "Black & White":
        filtered_image = apply_bw_filter(filtered_image)
    elif filter_type == "Sparkle":
        filtered_image = apply_sparkle_filter(filtered_image)
    elif filter_type == "Negative":
        filtered_image = apply_negative_filter(filtered_image)
    elif filter_type == "Glitch":
        filtered_image = apply_glitch_filter(filtered_image)
    
    result = cv2.bitwise_and(filtered_image, filtered_image, mask=mask)
    inverse_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(image, image, mask=inverse_mask)
    
    return cv2.add(result, background)

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    filtered_output = image.copy()
    selection_point = (-1, -1)

    hand_landmarks_list = results.multi_hand_landmarks if results.multi_hand_landmarks else []
    application_hand_landmarks = None
    selection_hand_landmarks = None
    
    if len(hand_landmarks_list) == 2:
        if hand_landmarks_list[0].landmark[0].x < hand_landmarks_list[1].landmark[0].x:
            selection_hand_landmarks = hand_landmarks_list[0]
            application_hand_landmarks = hand_landmarks_list[1]
        else:
            selection_hand_landmarks = hand_landmarks_list[1]
            application_hand_landmarks = hand_landmarks_list[0]
            
    elif len(hand_landmarks_list) == 1:
        selection_hand_landmarks = hand_landmarks_list[0]

    if selection_hand_landmarks:
        h, w, _ = image.shape
        tip_x = int(selection_hand_landmarks.landmark[TIP_IDS[1]].x * w)
        tip_y = int(selection_hand_landmarks.landmark[TIP_IDS[1]].y * h)
        selection_point = (tip_x, tip_y)

        cv2.circle(filtered_output, selection_point, 10, (0, 255, 0), -1)

        selected_filter = check_selection(selection_point)
        if selected_filter != "None":
            current_filter = selected_filter

    filtered_output = draw_ui(filtered_output, selection_point)
    
    if selection_hand_landmarks and application_hand_landmarks:
        filtered_output = apply_selective_filter(filtered_output, selection_hand_landmarks, application_hand_landmarks, current_filter)

    for hand_landmarks in hand_landmarks_list:
        mp_drawing.draw_landmarks(filtered_output, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Multi-Hand Filter Controller", filtered_output)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()