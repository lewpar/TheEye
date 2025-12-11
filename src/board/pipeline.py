import degirum
import cv2
import numpy

from degirum.model import Model
from devices import InferencingDeviceContext, InferencingDeviceType, get_device_model_context

from scipy.spatial.distance import cosine

detection_model: Model
embedding_model: Model

class BoundingBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    x1: int
    y1: int
    x2: int
    y2: int

class InferencingFaceData:
    def __init__(self, bounding_box: BoundingBox, score: int):
        self.bounding_box = bounding_box
        self.score = score

    bounding_box: BoundingBox
    score: float

class EmbeddingFaceData:
    similarity: float
    pass

def load_models():
    global detection_model, embedding_model

    device_type = InferencingDeviceType.CPU

    zoo = degirum.connect("@local", "./zoo")

    device: InferencingDeviceContext = get_device_model_context(device_type)

    detection_model = zoo.load_model(device.detection_model)
    embedding_model = zoo.load_model(device.embedding_model)

def start_camera() -> cv2.VideoCapture:
    """
    Starts the camera capture device. Throws an exception if the camera fails to open.
    """
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise Exception("Failed to start camera.")
    
    return camera

def get_camera_frame(camera: cv2.VideoCapture) -> cv2.typing.MatLike | None:
    """
    Gets the frame from the camera capture device and flips the frame horizontally.
    """
    result, frame = camera.read()

    if not result:
        return None
    
    return cv2.flip(frame, 1)

def get_faces_from_frame(frame: cv2.typing.MatLike) -> list[InferencingFaceData]:
    """
    Uses the yolov8n face detection model to find face bounding boxes and returns a list of all the faces.
    """
    inferencing_results = detection_model.predict(frame)
    faces = []

    for inference in inferencing_results.results:
        bounding_box = inference["bbox"]
        score = inference["score"]

        face = InferencingFaceData(
            BoundingBox(x1=int(bounding_box[0]), y1=int(bounding_box[1]), 
                        x2=int(bounding_box[2]), y2=int(bounding_box[3])),
            score)

        faces.append(face)

    return faces

def get_face_frame(frame: cv2.typing.MatLike, face_data: InferencingFaceData) -> cv2.typing.MatLike:
    """
    Takes face inferencing data and crops and resizes the frame to the size needed for running through the embedding model.
    """
    cropped_frame = frame[face_data.bounding_box.y1:face_data.bounding_box.y2, 
                          face_data.bounding_box.x1:face_data.bounding_box.x2]
    
    resized_frame = cv2.resize(cropped_frame, (112, 112))

    return resized_frame

def get_embedding_data(face_frame: cv2.typing.MatLike):
    result = embedding_model.predict(face_frame)
    for embed in result.results:
        return numpy.array(embed["data"]).flatten()