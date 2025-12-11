
import cv2
import pipeline

def main():
    pipeline.load_models()

    camera = pipeline.start_camera()

    while True:
        frame = pipeline.get_camera_frame(camera)

        if frame is None:
            print("Failed to get camera frame.")
            return
        
        faces = pipeline.get_faces_from_frame(frame)

        face_frames = []
        for face in faces:
            score = face.score
            top_left = (face.bounding_box.x1, face.bounding_box.y1)
            bottom_right = (face.bounding_box.x2, face.bounding_box.y2)

            cv2.putText(frame, f"Score: {score}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            face_frame = pipeline.get_face_frame(frame, face)
            face_frames.append(face_frame)

            embedding_data = pipeline.get_embedding_data(face_frame)
            #print(embedding_data)
    
        cv2.imshow("Face(s)", cv2.hconcat(face_frames))
        cv2.imshow("Camera", frame)
        cv2.pollKey()

if __name__ == "__main__":
    main()