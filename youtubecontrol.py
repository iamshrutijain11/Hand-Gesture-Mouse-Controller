import cv2
import mediapipe as mp
import pyautogui 
import time
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size() 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

SMOOTHING_FACTOR = 7
prev_mouse_x, prev_mouse_y = 0, 0 
last_click_time = 0
CLICK_DEBOUNCE_DELAY = 1.0 

MAPPING_BUFFER = 10 
CLICK_DISTANCE_THRESHOLD = 0.05
SCROLL_DISTANCE_THRESHOLD = 0.1 

print("Virtual YouTube Controller started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    is_clicking = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            index_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
            middle_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            ring_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
            
            finger_x = int(index_tip.x * frame_width)
            finger_y = int(index_tip.y * frame_height)
            
            index_up = index_tip.y < index_mcp_y
            middle_up = middle_tip.y < middle_mcp_y
            ring_up = ring_tip.y < ring_mcp_y 
            
            index_down = index_tip.y > index_mcp_y
            middle_down = middle_tip.y > middle_mcp_y
            
            distance_index_middle = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
            distance_index_ring = np.sqrt((index_tip.x - ring_tip.x)**2 + (index_tip.y - ring_tip.y)**2)
            is_three_finger_pinch = (
                index_up and middle_up and ring_up and 
                distance_index_middle < CLICK_DISTANCE_THRESHOLD and 
                distance_index_ring < 0.1 
            )

            if is_three_finger_pinch:
                is_clicking = True
                
                if (time.time() - last_click_time) > CLICK_DEBOUNCE_DELAY:
                    pyautogui.click() 
                    last_click_time = time.time()
                    
                    cv2.putText(frame, "CLICK (3-Finger Pinch)", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.circle(frame, (finger_x, finger_y), 15, (0, 0, 255), cv2.FILLED) 
                    
                continue 

            if distance_index_middle < SCROLL_DISTANCE_THRESHOLD:
                
                if index_up and middle_up:
                    pyautogui.scroll(5) 
                    cv2.putText(frame, "SCROLL UP (Fingers Up)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    continue 
                elif index_down and middle_down:
                    pyautogui.scroll(-5) 
                    cv2.putText(frame, "SCROLL DOWN (Fingers Down)", (10, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    continue 
            
            if index_up:
                
                cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 0), cv2.FILLED) 
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

                mouse_x = prev_mouse_x + (mouse_x - prev_mouse_x) / SMOOTHING_FACTOR
                mouse_y = prev_mouse_y + (mouse_y - prev_mouse_y) / SMOOTHING_FACTOR
                
                pyautogui.moveTo(mouse_x, mouse_y)
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
                
                cv2.putText(frame, "CURSOR MOVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if finger_x < frame_width * 0.2:
                    pyautogui.press('j') 
                    cv2.putText(frame, "SKIP BACK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    time.sleep(CLICK_DEBOUNCE_DELAY) 
                elif finger_x > frame_width * 0.8:
                    pyautogui.press('l') 
                    cv2.putText(frame, "SKIP FORWARD", (frame_width - 250, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    time.sleep(CLICK_DEBOUNCE_DELAY)

    cv2.imshow('Virtual Controller', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
