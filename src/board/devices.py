class InferencingDevice:
    def __init__(self, detection, embedding):
        self.detection_model = detection
        self.embedding_model = embedding

    detection_model: str
    embedding_model: str

INFERENCING_DEVICES = {
    "hailo": InferencingDevice("yolov8n_relu6_face--640x640_quant_hailort_multidevice_1", "arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1"),
    "cpu": InferencingDevice("yolov8n_relu6_face--640x640_quant_tflite_multidevice_1", "arcface_mobilefacenet--112x112_float_tflite_multidevice_1")
}