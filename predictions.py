import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from BaseModels import single_frame_details, frame_data, normal_landmarks, special_landmarks
import cv2
import mediapipe as mp

model_path = "holistic-custom-1024x1024-512-epoch-98.0%-0.1317.h5"

model = load_model(model_path)

required_frames = 30
frame_height = 1024
frame_width = 1024

classes = ['before', 'computer', 'cool', 'cousin', 'drink', 'go', 'help', 'inform', 'take', 'thin']


async def extract_keypoints(results : single_frame_details):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks]).flatten() if len(results.pose_landmarks) != 0 else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks]).flatten() if len(results.face_landmarks) !=0 else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks]).flatten() if len(results.left_hand_landmarks) !=0 else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks]).flatten() if len(results.right_hand_landmarks) !=0 else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


async def predict_sign_from_video(results : frame_data):
    if results is None:
        print("Predict Sign from Video : { Data is None } ")
        return None
    
    data = []
    for result in results.frame_data:
        data.append( await extract_keypoints(result) )

    data = np.array(data, dtype='float32')
    print(f"Input Shape Before : {data.shape}")
    data = np.expand_dims(data, axis=0)
    print(f"Input Shape After : {data.shape}")
    pred = model.predict(data)
    pred = classes[np.argmax(pred)]
    return  pred

# Function to process uploaded video and extract MediaPipe landmarks

async def process_uploaded_video(video_path: str) -> frame_data:
    """
    Process an uploaded video file and extract MediaPipe landmarks.
    Returns frame_data object containing landmark information for 30 frames.
    """
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic # type: ignore
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract (30 evenly spaced frames)
    if total_frames < required_frames:
        # If video has fewer frames than required, take all frames and pad with zeros
        frame_indices = list(range(total_frames))
    else:
        # Take evenly spaced frames across the video
        frame_indices = [int(i * total_frames / required_frames) for i in range(required_frames)]
    
    extracted_frames = []
    current_frame_index = 0
    
    while cap.isOpened() and len(extracted_frames) < required_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame_index in frame_indices:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = holistic.process(rgb_frame)
              # Extract landmarks and create frame data
            frame_landmarks = single_frame_details(
                frame_id=len(extracted_frames),
                face_landmarks=[
                    normal_landmarks(x=lm.x, y=lm.y, z=lm.z)
                    for lm in (results.face_landmarks.landmark if results.face_landmarks else [])
                ],
                pose_landmarks=[
                    special_landmarks(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
                    for lm in (results.pose_landmarks.landmark if results.pose_landmarks else [])
                ],
                left_hand_landmarks=[
                    normal_landmarks(x=lm.x, y=lm.y, z=lm.z)
                    for lm in (results.left_hand_landmarks.landmark if results.left_hand_landmarks else [])
                ],
                right_hand_landmarks=[
                    normal_landmarks(x=lm.x, y=lm.y, z=lm.z)
                    for lm in (results.right_hand_landmarks.landmark if results.right_hand_landmarks else [])
                ]
            )
            
            extracted_frames.append(frame_landmarks)
        
        current_frame_index += 1
    
    # If we have fewer frames than required, pad with empty frames
    while len(extracted_frames) < required_frames:
        empty_frame = single_frame_details(
            frame_id=len(extracted_frames),
            face_landmarks=[],
            pose_landmarks=[],
            left_hand_landmarks=[],
            right_hand_landmarks=[]
        )
        extracted_frames.append(empty_frame)
    
    # Clean up
    cap.release()
    holistic.close()
    
    # Return frame_data object
    return frame_data(frame_data=extracted_frames)

