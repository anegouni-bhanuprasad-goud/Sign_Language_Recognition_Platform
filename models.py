from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, DateTime
from datetime import datetime
from sqlalchemy.orm import relationship
from database import Base

class FrameData(Base):
    __tablename__ = "frame_data"
    id = Column(Integer, primary_key=True)
    prediction_label = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    frames = relationship("SingleFrameDetails", back_populates="frame_data", cascade="all, delete-orphan")


class SingleFrameDetails(Base):
    __tablename__ = "single_frame_details"
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, nullable=False)
    frame_data_id = Column(Integer, ForeignKey("frame_data.id"), nullable=False)

    frame_data = relationship("FrameData", back_populates="frames")
    face_landmarks = relationship("FaceLandmark", back_populates="frame", cascade="all, delete-orphan")
    pose_landmarks = relationship("PoseLandmark", back_populates="frame", cascade="all, delete-orphan")
    left_hand_landmarks = relationship("LeftHandLandmark", back_populates="frame", cascade="all, delete-orphan")
    right_hand_landmarks = relationship("RightHandLandmark", back_populates="frame", cascade="all, delete-orphan")


class FaceLandmark(Base):
    __tablename__ = "face_landmarks"
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    frame_id = Column(Integer, ForeignKey("single_frame_details.id"))
    frame = relationship("SingleFrameDetails", back_populates="face_landmarks")


class PoseLandmark(Base):
    __tablename__ = "pose_landmarks"
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    visibility = Column(Float)
    frame_id = Column(Integer, ForeignKey("single_frame_details.id"))
    frame = relationship("SingleFrameDetails", back_populates="pose_landmarks")


class LeftHandLandmark(Base):
    __tablename__ = "left_hand_landmarks"
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    frame_id = Column(Integer, ForeignKey("single_frame_details.id"))
    frame = relationship("SingleFrameDetails", back_populates="left_hand_landmarks")


class RightHandLandmark(Base):
    __tablename__ = "right_hand_landmarks"
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    frame_id = Column(Integer, ForeignKey("single_frame_details.id"))
    frame = relationship("SingleFrameDetails", back_populates="right_hand_landmarks")

