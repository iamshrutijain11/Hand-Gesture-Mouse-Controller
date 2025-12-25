import cv2
import mediapipe as mp
import keyboard
import time 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

zone_size = int(frame_width * 0.20)
LEFT_ZONE_MAX_X = zone_size
RIGHT_ZONE_MIN_X = frame_width - zone_size

last_action_time = 0
debounce_delay = 1.5 

print("Index Finger Gesture Control started. Press 'q' to exit.")
print("Move index finger to the LEFT edge for 'PREVIOUS'. Move to the RIGHT edge for 'NEXT'.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    current_time = time.time()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            finger_x = int(index_finger_tip.x * frame_width)
            finger_y = int(index_finger_tip.y * frame_height)
            
            cv2.circle(frame, (finger_x, finger_y), 15, (0, 255, 255), cv2.FILLED) 
            
            if finger_x < LEFT_ZONE_MAX_X and (current_time - last_action_time) > debounce_delay:
                print("Gesture Detected: PREVIOUS")
                keyboard.press_and_release('previous track')
                last_action_time = current_time 
                cv2.putText(frame, "PREVIOUS", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif finger_x > RIGHT_ZONE_MIN_X and (current_time - last_action_time) > debounce_delay:
                print("Gesture Detected: NEXT")
                keyboard.press_and_release('next track')
                last_action_time = current_time # Reset timer
                cv2.putText(frame, "NEXT", (frame_width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
    cv2.line(frame, (LEFT_ZONE_MAX_X, 0), (LEFT_ZONE_MAX_X, frame_height), (255, 0, 0), 2) 
    cv2.line(frame, (RIGHT_ZONE_MIN_X, 0), (RIGHT_ZONE_MIN_X, frame_height), (255, 0, 0), 2)

    cv2.imshow('Index Finger Controller', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
