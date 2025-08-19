from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, BackgroundTasks
from starlette.responses import FileResponse
from typing import Annotated
from datetime import datetime
import tempfile
import os
import time
from database import engine, SessionLocal
import models
from sqlalchemy.orm import Session
from BaseModels import frame_data

from predictions import predict_sign_from_video, process_uploaded_video

from fastapi.staticfiles import StaticFiles


app = FastAPI()

from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")



def createTable():
    models.Base.metadata.create_all(bind=engine)
    

createTable()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]


@app.get("/")
def index():
    return FileResponse("static\\index.html")

@app.get("/features")
def features():
    return FileResponse("static\\features.html")

@app.get("/about")
def about():
    return FileResponse("static\\about.html")


def save_data_to_db(db: Session, data: frame_data, prediction : str):
    try:
        start_time = datetime.utcnow()
        print(f"Saving to db... Started : {start_time}")
        
        frame_data_db = models.FrameData(prediction_label = prediction, timestamp = datetime.utcnow())

        for frame in data.frame_data:
            frame_db = models.SingleFrameDetails(
                frame_id=frame.frame_id
            )

            # Add face landmarks
            frame_db.face_landmarks = [
                models.FaceLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in frame.face_landmarks
            ]

            # Add pose landmarks
            frame_db.pose_landmarks = [
                models.PoseLandmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility) for lm in frame.pose_landmarks
            ]

            # Add left hand landmarks
            frame_db.left_hand_landmarks = [
                models.LeftHandLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in frame.left_hand_landmarks
            ]

            # Add right hand landmarks
            frame_db.right_hand_landmarks = [
                models.RightHandLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in frame.right_hand_landmarks
            ]

            # Add frame to parent
            frame_data_db.frames.append(frame_db)
        
        db.add(frame_data_db)
        db.commit()
        db.refresh(frame_data_db)

        end_time = datetime.utcnow()
        print(f"Saved Data to Database... End Time : {end_time}, Time Taken : {end_time - start_time}")

    except Exception as e:
        db.rollback()
        print(f"Error saving data to database in background: {e}")

@app.post("/predict")
async def predict(db : db_dependency, background_tasks : BackgroundTasks, data : frame_data):

    start_time = datetime.utcnow()
    print("Data Arrived : ", start_time)

    prediction = await predict_sign_from_video(data)

    curr_time = datetime.utcnow()
    print("Data Arrived from Prediction : ", curr_time, "\nTime taken for Prediction : ", curr_time - start_time)

    background_tasks.add_task(save_data_to_db, db=db, data=data, prediction=prediction)

    end_time = datetime.utcnow()
    print("Data Sending to frontend: ", end_time, "\nTime Taken for Process :", end_time - start_time)

    response = { "prediction" : f"Resnet Model Predicted : {prediction}"}
    time.sleep(10)

    return response

@app.get("/all_records")
async def all_records(db : db_dependency):
    records = db.query(models.FrameData).all()
    message = []
    for record in records:
        id = record.id
        result = await get_single_record(id, db)
        message.append(result)
    return message


async def get_single_record(id, db : db_dependency):
    record = db.query(models.FrameData).filter(models.FrameData.id == id).first()
    frames = db.query(models.SingleFrameDetails).filter(models.SingleFrameDetails.frame_data_id == record.id).all() if record else None
    face_landmarks = [db.query(models.FaceLandmark).filter(models.FaceLandmark.frame_id == frame.id).all() for frame in frames] if frames else None
    pose_landmarks = [db.query(models.PoseLandmark).filter(models.PoseLandmark.frame_id == frame.id).all() for frame in frames] if frames else None
    left_hand_landmarks = [db.query(models.LeftHandLandmark).filter(models.LeftHandLandmark.frame_id == frame.id).all() for frame in frames] if frames else None
    right_hand_landmarks = [db.query(models.RightHandLandmark).filter(models.RightHandLandmark.frame_id == frame.id).all() for frame in frames] if frames else None

    frame_data = []
    if frames:
        for i, frame in enumerate(frames):
            face_landmarks_data = []
            pose_landmarks_data = []
            left_hand_landmarks_data = []
            right_hand_landmarks_data  = []

            # FACE
            if face_landmarks:
                for landmark in face_landmarks[i]:  # Get the list of landmarks for this frame
                    face_landmarks_data.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })

            # POSE
            if pose_landmarks:
                for landmark in pose_landmarks[i]:
                    pose_landmarks_data.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility  # Also fix typo: `visilibility` → `visibility`
                    })

            # LEFT HAND
            if left_hand_landmarks:
                for landmark in left_hand_landmarks[i]:
                    left_hand_landmarks_data.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })

            # RIGHT HAND
            if right_hand_landmarks:
                for landmark in right_hand_landmarks[i]:
                    right_hand_landmarks_data.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })

            frame_data.append({
                "frame_id": frame.frame_id,
                "face_landmarks": face_landmarks_data,
                "pose_landmarks": pose_landmarks_data,
                "left_hand_landmarks": left_hand_landmarks_data,
                "right_hand_landmarks": right_hand_landmarks_data
            })


    message = {}
    if record:
        message = {
            "id" : record.id,
            "prediction_label" : record.prediction_label,
            "timestamp" : record.timestamp,
            "frames" : frame_data
        }

    return message if len(message) != 0 else {"message" : f"No record found with id : {id}"}

@app.get("/record")
async def get_record(id: int, db: db_dependency):
    return get_single_record(id, db)

@app.get("/delete_record")
async def delete_record(db: db_dependency, id : int):
    try:
        query = db.query(models.FrameData).filter(models.FrameData.id == id).first()
        if not query:
            return {'message' : f"No record found with id : {id}"}
        db.delete(query)
        db.commit()
      
        return {"message" : "Record deleted"}
    
    except Exception as e:
        return {"message" : str(e)}

@app.get("/reset")
async def reset_db(db : db_dependency):
    try:
        models.Base.metadata.drop_all(bind = engine)

        createTable()

        return {"message" : "Database reset successfully"}
    except Exception as e:
        return {"message" : str(e)}
    

    # Endpoint to handle video file uploads for sign language prediction.

@app.post("/predict-video")
async def predict_video_upload(db: db_dependency, video: UploadFile = File(...)):
    """
    Endpoint to handle video file uploads for sign language prediction.
    Processes the video file and returns prediction results.
    """
    try:
        # Validate file type
        if not video.content_type.startswith('video/'): # type: ignore
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
        
        # Validate file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB
        contents = await video.read()
        if len(contents) > max_size:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as temp_file: # type: ignore
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        try:
            # Process video and extract frame data
            video_frame_data = await process_uploaded_video(temp_file_path)
            
            if not video_frame_data or not video_frame_data.frame_data:
                raise HTTPException(status_code=400, detail="Could not extract valid frames from video.")
            
            # Make prediction
            prediction = await predict_sign_from_video(video_frame_data)
            
            if not prediction:
                raise HTTPException(status_code=500, detail="Prediction failed.")
            
            # Save to database
            frame_data_db = models.FrameData(
                prediction_label=prediction, 
                timestamp=datetime.utcnow()
            )

            for frame in video_frame_data.frame_data:
                frame_db = models.SingleFrameDetails(
                    frame_id=frame.frame_id
                )

                # Add face landmarks
                frame_db.face_landmarks = [
                    models.FaceLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in frame.face_landmarks
                ]

                # Add pose landmarks
                frame_db.pose_landmarks = [
                    models.PoseLandmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility) for lm in frame.pose_landmarks
                ]

                # Add left hand landmarks
                frame_db.left_hand_landmarks = [
                    models.LeftHandLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in frame.left_hand_landmarks
                ]

                # Add right hand landmarks
                frame_db.right_hand_landmarks = [
                    models.RightHandLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in frame.right_hand_landmarks
                ]

                # Add frame to parent
                frame_data_db.frames.append(frame_db)
            
            db.add(frame_data_db)
            db.commit()
            db.refresh(frame_data_db)

            response = {"prediction": f"Model Predicted : {prediction}"}
            return response
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
