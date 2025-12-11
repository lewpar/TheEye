
import cv2
import numpy

import pipeline

from scipy.spatial.distance import cosine

def main():
    pipeline.load_models()

    camera = pipeline.start_camera()

    while True:
        frame = pipeline.get_camera_frame(camera)

        if frame is None:
            print("Failed to get camera frame.")
            return
        
        faces = pipeline.get_faces_from_frame(frame)

        for face in faces:
            score = face.score
            top_left = (face.bounding_box.x1, face.bounding_box.y1)
            bottom_right = (face.bounding_box.x2, face.bounding_box.y2)

            cv2.putText(frame, f"Score: {score}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            face_frame = pipeline.get_face_frame(frame, face)
            cv2.imshow("Face Frame", face_frame)

            # embedding_result = embedding.predict(resized_frame)
            # for embed in embedding_result.results:
            #     # Get the vectorized embedding data and flatten it to a 1d array
            #     embed_data = numpy.array(embed["data"]).flatten()

            #     if len(previous_face) < 1:
            #         previous_face = embed_data
            #         continue

            #     # Compare the vectorized data to see if they are near each other,
            #     # must pass in a flattened vector
            #     cosine_sim = 1 - cosine(embed_data, previous_face)

            #     top_left_inner = (int(top_left[0]), int(top_left[1]) + 20)
            #     cv2.putText(frame, f"Score: {cosine_sim}", top_left_inner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    
        cv2.imshow("Camera", frame)
        cv2.pollKey()

if __name__ == "__main__":
    main()