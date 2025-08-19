from pydantic import BaseModel
from typing import List


class normal_landmarks(BaseModel):
    x: float
    y: float
    z: float

class special_landmarks(BaseModel):
    x: float
    y: float
    z: float
    visibility : float

class single_frame_details(BaseModel):
    frame_id              :   int
    face_landmarks        :   List[normal_landmarks] 
    pose_landmarks        :   List[special_landmarks]
    left_hand_landmarks   :   List[normal_landmarks]
    right_hand_landmarks  :   List[normal_landmarks]

class frame_data(BaseModel):
    frame_data : List[single_frame_details]