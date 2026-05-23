import cv2




class FaceDetector:


    def __init__(self):
        
        self.face_detection_model = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )


    def detect_faces(self, frame):

        faces = self.face_detection_model.detectMultiScale(
            frame,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (50,50)
        )
        if len(faces) > 0:
            cv2.imwrite(f"screenshot.jpg", frame)
        return faces
    

    def draw_rectangles(self, frame, x, y, w, h):

        cv2.rectangle(
            frame,
            (x,y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )
