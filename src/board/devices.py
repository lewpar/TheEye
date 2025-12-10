from enum import Enum

class InferencingDeviceType(Enum):
    CPU = 1,
    HAILO = 2

class InferencingDeviceContext:
    def __init__(self, device_type, detection_model, embedding_model):
        self.device_type = device_type
        self.detection_model = detection_model
        self.embedding_model = embedding_model

    device_type: InferencingDeviceType
    detection_model: str
    embedding_model: str

def get_device_model_context(device_type: InferencingDeviceType) -> InferencingDeviceContext:
    if device_type == InferencingDeviceType.CPU:
        return InferencingDeviceContext(device_type=InferencingDeviceType.CPU,
                                        detection_model="yolov8n_relu6_face--640x640_quant_tflite_multidevice_1",
                                        embedding_model="arcface_mobilefacenet--112x112_float_tflite_multidevice_1")
    elif device_type == InferencingDeviceType.HAILO:
        return InferencingDeviceContext(device_type=InferencingDeviceType.HAILO,
                                     detection_model="yolov8n_relu6_face--640x640_quant_hailort_multidevice_1",
                                     embedding_model="arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1")
