import degirum
import cv2

import devices

from scipy.spatial.distance import cosine

def main():
    # TODO: Move this to an environment variable
    device_type = devices.InferencingDeviceType.CPU

    zoo = degirum.connect("@local", "./zoo")

    device = devices.get_device_model_context(device_type)

    detection = zoo.load_model(device.detection_model)
    embedding = zoo.load_model(device.embedding_model)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Failed to open camera stream.")
        return
    
    previous_face = []

    while True:
        result, frame = camera.read()

        if not result:
            print("Failed to get frame from camera.")
            return
        
        frame = cv2.flip(frame, 1)

        inferencing_results = detection.predict(frame)
        for inference in inferencing_results.results:
            bounding_box = inference["bbox"]
            score = inference["score"]

            x1 = int(bounding_box[0])
            y1 = int(bounding_box[1])

            x2 = int(bounding_box[2])
            y2 = int(bounding_box[3])

            top_left = (x1, y1)
            bottom_right = (x2, y2)

            score_clean = int(score * 100)

            cv2.putText(frame, f"Score: {score_clean}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            cropped_frame = frame[y1:y2, x1:x2]
            cv2.imshow("Cropped Frame", cropped_frame)

            resized_frame = cv2.resize(cropped_frame, (112, 112))
            cv2.imshow("Resized Frame", resized_frame)

            embedding_result = embedding.predict(resized_frame)
            for embed in embedding_result.results:
                embed_data = embed["data"].flatten()

                if len(previous_face) < 1:
                    previous_face = embed_data
                    continue

                cosine_sim = 1 - cosine(embed_data, previous_face)

                print(cosine_sim)
    
        cv2.imshow("Camera", frame)
        cv2.pollKey()

if __name__ == "__main__":
    main()