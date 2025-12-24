import cv2
import mediapipe as mp
import pyautogui # For controlling the mouse and sending clicks
import time
import numpy as np
# --- MEDIA PIPE SETUP ---
mp_hands = mp.solutions.hands
# Use static_image_mode=False for video
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- PYAUTOGUI SETUP ---
# Get screen dimensions to map hand movement to cursor movement
screen_width, screen_height = pyautogui.size() 

# --- OPENCV & CONTROL SETUP ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Control parameters for smoothing and mapping
SMOOTHING_FACTOR = 7
prev_mouse_x, prev_mouse_y = 0, 0 

# Debounce timer for click action
last_click_time = 0
CLICK_DEBOUNCE_DELAY = 1.0 # Seconds between clicks

# --- MAPPING REFINEMENT ---
MAPPING_BUFFER = 10 
# Click distance threshold (must be low for a 'pinch')
CLICK_DISTANCE_THRESHOLD = 0.05
# Scroll distance threshold (can be higher for a more relaxed 'two-finger' gesture)
SCROLL_DISTANCE_THRESHOLD = 0.1 

print("Virtual YouTube Controller started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame for a mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # --- DEFAULT CONTROL ACTION ---
    is_clicking = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw the hand skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get key landmarks
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            # NEW: Ring Finger Tip
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            # Get the y-coordinates of the MCP (knuckles)
            index_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
            middle_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            # NEW: Ring Finger MCP
            ring_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
            
            # Convert normalized index tip coordinates to pixel coordinates
            finger_x = int(index_tip.x * frame_width)
            finger_y = int(index_tip.y * frame_height)
            
            # --- GESTURE STATES ---
            # Finger Up (tip Y < knuckle Y)
            index_up = index_tip.y < index_mcp_y
            middle_up = middle_tip.y < middle_mcp_y
            ring_up = ring_tip.y < ring_mcp_y # NEW
            
            # Finger Down (tip Y > knuckle Y)
            index_down = index_tip.y > index_mcp_y
            middle_down = middle_tip.y > middle_mcp_y
            
            # Distance between Index and Middle tips
            distance_index_middle = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
            # NEW: Distance for 3-finger click check (using Index and Ring tips as anchor points)
            distance_index_ring = np.sqrt((index_tip.x - ring_tip.x)**2 + (index_tip.y - ring_tip.y)**2)
            
            # --- GESTURE RECOGNITION ---
            
            # 1. CLICK/PLAY/PAUSE (Three-Finger Gesture: Index, Middle, Ring up and close together)
            # This check is prioritized to ensure it executes clearly when the gesture is made.
            
            # Check if all three are up AND they are close to each other
            is_three_finger_pinch = (
                index_up and middle_up and ring_up and # All three fingers are pointing up
                distance_index_middle < CLICK_DISTANCE_THRESHOLD and # Index and Middle are close
                distance_index_ring < 0.1 # Index and Ring are reasonably close to form a group
            )

            if is_three_finger_pinch:
                is_clicking = True
                
                if (time.time() - last_click_time) > CLICK_DEBOUNCE_DELAY:
                    pyautogui.click() 
                    last_click_time = time.time()
                    
                    cv2.putText(frame, "CLICK (3-Finger Pinch)", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.circle(frame, (finger_x, finger_y), 15, (0, 0, 255), cv2.FILLED) # Red indicator for click
                    
                continue # Skip remaining gesture checks

            
            # 2. SCROLL GESTURE (Index and Middle fingers together, up or down)
            # This check is prioritized over movement to keep scrolling fluid.
            if distance_index_middle < SCROLL_DISTANCE_THRESHOLD:
                
                # SCROLL UP: Both Index and Middle fingers are UP
                if index_up and middle_up:
                    pyautogui.scroll(5) # Scroll up
                    cv2.putText(frame, "SCROLL UP (Fingers Up)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    continue # Skip remaining gesture checks
                    
                # SCROLL DOWN: Both Index and Middle fingers are DOWN (curled)
                elif index_down and middle_down:
                    pyautogui.scroll(-5) # Scroll down
                    cv2.putText(frame, "SCROLL DOWN (Fingers Down)", (10, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    continue # Skip remaining gesture checks
            
            
            # 3. CURSOR MOVEMENT & KEY COMMANDS (Single Index Finger Up)
            if index_up:
                
                # Draw a visual indicator for the index finger tip
                cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 0), cv2.FILLED) 

                # --- MOUSE MOVEMENT (Cursor Control) ---
                
                # Map finger position in webcam frame to screen coordinates
                mouse_x = np.interp(
                    finger_x, 
                    (MAPPING_BUFFER, frame_width - MAPPING_BUFFER), 
                    (0, screen_width)
                )
                mouse_y = np.interp(
                    finger_y, 
                    (MAPPING_BUFFER, frame_height - MAPPING_BUFFER), 
                    (0, screen_height)
                )

                # Apply smoothing
                mouse_x = prev_mouse_x + (mouse_x - prev_mouse_x) / SMOOTHING_FACTOR
                mouse_y = prev_mouse_y + (mouse_y - prev_mouse_y) / SMOOTHING_FACTOR
                
                pyautogui.moveTo(mouse_x, mouse_y)
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
                
                cv2.putText(frame, "CURSOR MOVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


                # --- YOUTUBE SPECIFIC KEY COMMANDS (Index Finger in Zones) ---
                
                # LEFT/RIGHT Commands (YouTube video skip/speed control)
                if finger_x < frame_width * 0.2:
                    pyautogui.press('j') # YouTube: Skip backward 10 seconds
                    cv2.putText(frame, "SKIP BACK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    time.sleep(CLICK_DEBOUNCE_DELAY) 
                elif finger_x > frame_width * 0.8:
                    pyautogui.press('l') # YouTube: Skip forward 10 seconds
                    cv2.putText(frame, "SKIP FORWARD", (frame_width - 250, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    time.sleep(CLICK_DEBOUNCE_DELAY)


    # Show the webcam feed
    cv2.imshow('Virtual Controller', frame)
    
    # Exit condition
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()