import cv2
import asyncio
import websockets
import struct
import pipeline

# --- Global state ---
clients = set()
latest_packet = None       # stores the most recent JPEG packet
new_frame_event = asyncio.Event()  # used to notify clients a new frame is ready


# -----------------------------
#  Broadcaster: ONE camera loop
# -----------------------------
async def camera_broadcaster():
    print("Loading models...")
    pipeline.load_models()
    print("Starting camera...")
    camera = pipeline.start_camera()

    while True:
        frame = pipeline.get_camera_frame(camera)
        if frame is None:
            print("Failed to get camera frame.")
            await asyncio.sleep(0.01)
            continue

        # Face detection
        faces = pipeline.get_faces_from_frame(frame)

        face_frames = []
        for face in faces:
            score = face.score
            top_left = (face.bounding_box.x1, face.bounding_box.y1)
            bottom_right = (face.bounding_box.x2, face.bounding_box.y2)

            cv2.putText(frame, f"Score: {score}", top_left,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            face_frame = pipeline.get_face_frame(frame, face)
            face_frames.append(face_frame)

        # Handle no faces (cv2.hconcat would crash)
        if len(face_frames) > 0:
            faces_img = cv2.hconcat(face_frames)
        else:
            faces_img = frame

        ok, jpeg = cv2.imencode(".jpg", frame)
        image_bytes = jpeg.tobytes()

        packet = image_bytes  # already bytes

        # Save packet globally
        global latest_packet
        latest_packet = packet

        # Notify all clients
        new_frame_event.set()

        # Reset the event so clients wait for next frame
        new_frame_event.clear()

        # Debug preview (optional)
        cv2.imshow("Camera", frame)
        cv2.imshow("Faces", faces_img)
        cv2.pollKey()

        await asyncio.sleep(0)  # yield to event loop


# -----------------------------
#  Client handler (no camera!)
# -----------------------------
async def handler(ws):
    # Register client
    clients.add(ws)
    print("Client connected. Total:", len(clients))

    try:
        # If there's already a frame, send it immediately
        if latest_packet is not None:
            await ws.send(latest_packet)

        # Wait for new frames and send them
        while True:
            await new_frame_event.wait()
            if latest_packet is not None:
                await ws.send(latest_packet)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

    finally:
        clients.remove(ws)
        print("Client removed. Total:", len(clients))


# -----------------------------
#  Main
# -----------------------------
async def main():
    # Start camera broadcaster as background task
    asyncio.create_task(camera_broadcaster())

    # Start websocket server
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("Server ready...")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
