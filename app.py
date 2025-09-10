import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.set_page_config(page_title="Realtime Hand Filter Demo", layout="centered")
st.title("🖐️ Real-Time Hand Filter App")
st.markdown(
    """
    <style>
    .css-1r6slb0 {background: #393e46;}
    .main {background: #23272f;}
    h1, h2, footer {color: #ffd369;}
    .stButton>button {background-color: #ffd369; color:black;}
    </style>
    """, unsafe_allow_html=True
)
st.markdown(
    "**Use your hands to select filters! Point with your index finger to choose a filter, then use both hands to create the filter area.**"
)

TIP_IDS = [4, 8, 12, 16, 20]
FILTERS = ["Black & White", "Sparkle", "Negative", "Glitch", "Spotlight"]

# ---- Helper Functions (filters) ----
def apply_bw_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

def apply_sparkle_filter(image):
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
    return cv2.bitwise_not(image)

def apply_glitch_filter(image):
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

def draw_ui(image, selection_point, filter_rects):
    """Draws the filter selection menu with button highlights."""
    filter_rects.clear()  # Clear previous rectangles
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

        if x1 < selection_point[0] < x2 and y1 < selection_point[1] < y2:
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 255, 100), -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        text_size = cv2.getTextSize(filter_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (button_width - text_size[0]) // 2
        text_y = y1 + (menu_height + text_size[1]) // 2
        cv2.putText(image, filter_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image

def check_selection(point, filter_rects):
    """Checks if the given point is within any of the filter UI rectangles."""
    for i, rect in enumerate(filter_rects):
        x1, y1, x2, y2 = rect
        if x1 < point[0] < x2 and y1 < point[1] < y2:
            print(f"DEBUG: Filter {FILTERS[i]} selected at point {point} in rect {rect}")  # Debug
            return FILTERS[i]
    print(f"DEBUG: No filter selected at point {point}, checked {len(filter_rects)} rects")  # Debug
    return "None"

def apply_spotlight_filter(image, mask):
    """Makes the area between the hands visible and the rest of the screen black."""
    black_screen = np.zeros_like(image)
    return cv2.bitwise_and(image, image, mask=mask) + cv2.bitwise_and(black_screen, black_screen, mask=cv2.bitwise_not(mask))

def apply_selective_filter(image, landmarks1, landmarks2, filter_type):
    """
    Applies the selected filter to the area between the thumb and index finger
    of both hands, creating a precise quadrilateral window.
    """
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    mp_hands = mp.solutions.hands

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
    else:
        # If no filter matches, return original
        return image
    
    result = cv2.bitwise_and(filtered_image, filtered_image, mask=mask)
    inverse_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(image, image, mask=inverse_mask)
    
    return cv2.add(result, background)

class HandFilterTransformer(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        self.current_filter = "None"
        self.filter_rects = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        filtered_output = img.copy()
        selection_point = (-1, -1)

        hand_landmarks_list = results.multi_hand_landmarks if results.multi_hand_landmarks else []
        application_hand_landmarks = None
        selection_hand_landmarks = None
        
        # Hand assignment logic from Main.py
        if len(hand_landmarks_list) == 2:
            if hand_landmarks_list[0].landmark[0].x < hand_landmarks_list[1].landmark[0].x:
                selection_hand_landmarks = hand_landmarks_list[0]
                application_hand_landmarks = hand_landmarks_list[1]
            else:
                selection_hand_landmarks = hand_landmarks_list[1]
                application_hand_landmarks = hand_landmarks_list[0]
        elif len(hand_landmarks_list) == 1:
            selection_hand_landmarks = hand_landmarks_list[0]

        # Get selection finger position
        if selection_hand_landmarks:
            h, w, _ = img.shape
            tip_x = int(selection_hand_landmarks.landmark[TIP_IDS[1]].x * w)
            tip_y = int(selection_hand_landmarks.landmark[TIP_IDS[1]].y * h)
            selection_point = (tip_x, tip_y)

            # Draw green circle on selection finger
            cv2.circle(filtered_output, selection_point, 10, (0, 255, 0), -1)

        # Draw interactive UI overlay FIRST to populate filter_rects
        filtered_output = draw_ui(filtered_output, selection_point, self.filter_rects)
        
        # Check for filter selection AFTER UI is drawn
        if selection_hand_landmarks and len(self.filter_rects) > 0:
            selected_filter = check_selection(selection_point, self.filter_rects)
            if selected_filter != "None":
                self.current_filter = selected_filter
                # Visual feedback for successful selection
                cv2.putText(filtered_output, f"SELECTED: {selected_filter}!", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display current filter and debugging info at the top
        cv2.putText(filtered_output, f"Current Filter: {self.current_filter}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(filtered_output, f"Selection Point: {selection_point}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(filtered_output, f"Filter Rects: {len(self.filter_rects)}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Apply filter between both hands if available
        if selection_hand_landmarks and application_hand_landmarks and self.current_filter != "None":
            filtered_output = apply_selective_filter(
                filtered_output, 
                selection_hand_landmarks, 
                application_hand_landmarks, 
                self.current_filter
            )

        # Draw hand landmarks
        for hand_landmarks in hand_landmarks_list:
            self.drawing.draw_landmarks(filtered_output, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return av.VideoFrame.from_ndarray(filtered_output, format="bgr24")

st.header("Live Camera Feed & Filter Demo")
st.info("📋 **Instructions:**\n"
        "1. Show both hands to the camera\n"
        "2. Point with your index finger to select a filter from the bottom menu\n"
        "3. Create a quadrilateral area with thumb and index finger tips of both hands\n"
        "4. The selected filter will be applied to the area between your hands!")

# Add current filter display
st.sidebar.markdown("### Current Filter")
if 'current_filter' not in st.session_state:
    st.session_state.current_filter = "None"

webrtc_streamer(
    key="hand-filter-demo",
    video_processor_factory=HandFilterTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.sidebar.markdown("---")
st.sidebar.info("🎯 **Tips:**\n"
               "• Use your left hand (on your right in mirrored view) for selection\n"
               "• Point at the filter buttons with your index finger\n"
               "• Create a window with both hands' thumb and index tips\n"
               "• Move hands to adjust the filter area size")

st.sidebar.markdown("---")
st.sidebar.info("Created with ❤️ using OpenCV, Streamlit, MediaPipe, and streamlit-webrtc.")

# Instructions for deployment (optional)
st.markdown("""
<details>
<summary><b>How to run locally?</b></summary>

- Install requirements: `pip install streamlit opencv-python mediapipe streamlit-webrtc numpy`
- Run the app: `streamlit run app.py`
- Open the shown localhost link in browser!
</details>
""", unsafe_allow_html=True)
