# it is 90% AI program don't expect much from it

import cv2
import numpy as np
import tensorflow as tf
import speech_recognition as sr
import requests
import random
import time
from threading import Thread
import os
import pickle
from sklearn import neighbors
from sklearn.metrics.pairwise import cosine_similarity

class VisionSystem:
    def __init__(self, left_camera_id=0, right_camera_id=1, model_path='path/to/saved_model', fov=60):
        self.cap_left = cv2.VideoCapture(left_camera_id)
        self.cap_right = cv2.VideoCapture(right_camera_id)
        self.model = tf.saved_model.load(model_path)
        self.fov = fov

        # Load face detection model (Haar cascade for fast detection)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = []  # List to store known face features
        self.known_names = []  # List to store corresponding names
        self.face_features = []  # List to store features of known faces

        # Load existing model if available
        self.face_recognition_model = self.load_face_recognition_model()

    def generate_depth_map(self, frame_left, frame_right):
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(gray_left, gray_right)
        depth_map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)
        return depth_map

    def calculate_position(self, x, y, z, width=640, height=480):
        x_angle = (x - width // 2) * (self.fov / width)
        y_angle = (y - height // 2) * (self.fov / height)
        x_real = z * np.tan(np.radians(x_angle))
        y_real = z * np.tan(np.radians(y_angle))
        return x_real, y_real, z

    def detect_objects(self, frame_left, depth_map):
        input_tensor = tf.convert_to_tensor([frame_left], dtype=tf.uint8)
        detections = self.model(input_tensor)

        detected_objects = []
        for detection in detections['detection_boxes']:
            x, y = int(detection[1] * frame_left.shape[1]), int(detection[0] * frame_left.shape[0])
            z = depth_map[y, x]

            distance = z  # Assuming z is in cm after calibration
            takeable = distance < 38

            position = self.calculate_position(x, y, z)
            detected_objects.append((detection['detection_classes'], position, takeable))

        return detected_objects

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def process_frames(self):
        while True:
            ret_left, frame_left = self.cap_left.read()
            ret_right, frame_right = self.cap_right.read()

            if ret_left and ret_right:
                depth_map = self.generate_depth_map(frame_left, frame_right)
                detected_objects = self.detect_objects(frame_left, depth_map)

                faces = self.detect_faces(frame_left)
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame_left, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        face_img = frame_left[y:y+h, x:x+w]
                        self.recognize_or_register_face(face_img)
                    print("Face detected!")
                else:
                    print(detected_objects)
                
                cv2.imshow('Depth Map', depth_map)
                cv2.imshow('Frame Left', frame_left)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap_left.release()
        self.cap_right.release()
        cv2.destroyAllWindows()

    def recognize_or_register_face(self, face_img):
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_feature = gray_face.flatten()  # Flatten the face image to a 1D vector

        if self.known_faces:
            # Calculate similarity with known faces
            similarities = [cosine_similarity([face_feature], [known_face])[0][0] for known_face in self.face_features]
            max_similarity = max(similarities, default=0)
            
            if max_similarity > 0.77:
                print("Face recognized with high similarity!")
                return
        
        # Register new face
        self.register_new_face(face_img)

    def register_new_face(self, face_img):
        face_dir = 'known_faces'
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)

        # Take multiple pictures of the new face
        for i in range(5):  # Take 5 pictures
            face_filename = os.path.join(face_dir, f"face_{time.time()}.png")
            cv2.imwrite(face_filename, face_img)
            time.sleep(2)  # Wait for 2 seconds between pictures
        
        # Request the person's name using speech recognition
        name = self.request_name_via_speech()
        if not name:
            print("Failed to get name.")
            return
        
        # Save the face features
        self.face_features.append(face_img.flatten())
        self.known_names.append(name)
        
        # Save the updated face recognition model
        self.save_face_recognition_model()

    def request_name_via_speech(self):
        # Prompt the person to say their name
        print("Please say your name.")
        # Code to use text-to-speech to prompt the person
        # Assuming the robot has a way to speak
        # For now, simulate this with a print statement
        
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            audio = recognizer.listen(source, timeout=10)
            try:
                name = recognizer.recognize_google(audio)
                print(f"Heard name: {name}")
                return name
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand that.")
            except sr.RequestError as e:
                print(f"Request error from Google Speech Recognition service; {e}")
        return None

    def load_face_recognition_model(self):
        try:
            with open('face_recognition_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.face_features = model_data.get('face_features', [])
                self.known_names = model_data.get('known_names', [])
                return neighbors.KNeighborsClassifier()  # Use KNN classifier
        except FileNotFoundError:
            return neighbors.KNeighborsClassifier()  # Return a new model if not found

    def save_face_recognition_model(self):
        with open('face_recognition_model.pkl', 'wb') as f:
            model_data = {
                'face_features': self.face_features,
                'known_names': self.known_names
            }
            pickle.dump(model_data, f)

class ServoControl:
    def __init__(self):
        self.servo_x = 90  # Default angle for x-axis (center)
        self.servo_y = 90  # Default angle for y-axis (center)

    def move_servo(self, servo, angle):
        # This is a placeholder function
        if servo == "x":
            self.servo_x = angle
        elif servo == "y":
            self.servo_y = angle
        print(f"Moving servo {servo} to angle {angle}")

    def aim_at_object(self, target_x, target_y):
        while True:
            current_x = self.servo_x
            current_y = self.servo_y

            if abs(target_x - current_x) < 1 and abs(target_y - current_y) < 1:
                break  # Object is centered

            if target_x < current_x:
                current_x -= 1
            elif target_x > current_x:
                current_x += 1

            if target_y < current_y:
                current_y -= 1
            elif target_y > current_y:
                current_y += 1

            self.move_servo("x", current_x)
            self.move_servo("y", current_y)
            time.sleep(0.05)  # Small delay for smoother movement

    def aim_at_face(self, face):
        (x, y, w, h) = face
        x_center = x + w // 2
        y_center = y + h // 2
        self.aim_at_object(x_center, y_center)

class SpeechRecognitionSystem:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def listen_and_recognize(self):
        with sr.Microphone() as source:
            while True:
                print("Listening for sound...")
                audio = self.recognizer.listen(source)
                if self.recognizer.energy_threshold < audio.rms:
                    try:
                        text = self.recognizer.recognize_google(audio)
                        print(f"Detected speech: {text}")
                        return text
                    except sr.UnknownValueError:
                        print("Sorry, I couldn't understand that.")
                    except sr.RequestError as e:
                        print(f"Request error from Google Speech Recognition service; {e}")

class RoboticBrain:
    def __init__(self):
        self.vision_system = VisionSystem()
        self.speech_recognition_system = SpeechRecognitionSystem()
        self.servo_control = ServoControl()
        self.object_lock = False  # Lock to check if we're locked onto a face

    def make_api_request(self):
        while True:
            ret_left, frame_left = self.vision_system.cap_left.read()
            if ret_left:
                depth_map = self.vision_system.generate_depth_map(frame_left, None)
                detected_objects = self.vision_system.detect_objects(frame_left, depth_map)
                faces = self.vision_system.detect_faces(frame_left)

                visible_objects = [obj[0] for obj in detected_objects]
                takeable_objects = [obj for obj in detected_objects if obj[2]]
                is_person = len(faces) > 0

                person_speech = ""
                if is_person:
                    person_speech = self.speech_recognition_system.listen_and_recognize()

                api_payload = {
                    "Visible object": visible_objects,
                    "Takeable object": takeable_objects,
                    "Is there person": is_person,
                    "Person detected speech": person_speech
                }

                try:
                    response = requests.post('api_url', json=api_payload)
                    response.raise_for_status()
                    print("API Response:", response.status_code, response.json())
                except requests.RequestException as e:
                    print(f"API request error: {e}")

            time.sleep(4)

    def random_head_movement(self):
        while True:
            if not self.object_lock:
                sleep_time = random.uniform(20, 90)
                time.sleep(sleep_time)

                ret_left, frame_left = self.vision_system.cap_left.read()
                if ret_left:
                    depth_map = self.vision_system.generate_depth_map(frame_left, None)
                    detected_objects = self.vision_system.detect_objects(frame_left, depth_map)
                    
                    if detected_objects:
                        selected_object = random.choice(detected_objects)
                        x_angle, y_angle, _ = selected_object[1]
                        self.servo_control.aim_at_object(x_angle, y_angle)

    def track_face(self):
        while True:
            ret_left, frame_left = self.vision_system.cap_left.read()
            if ret_left:
                faces = self.vision_system.detect_faces(frame_left)
                if len(faces) > 0:
                    self.object_lock = True
                    for face in faces:
                        self.servo_control.aim_at_face(face)
                    print("Tracking face!")
                else:
                    self.object_lock = False

    def run(self):
        api_request_thread = Thread(target=self.make_api_request)
        face_tracking_thread = Thread(target=self.track_face)
        head_movement_thread = Thread(target=self.random_head_movement)

        api_request_thread.start()
        face_tracking_thread.start()
        head_movement_thread.start()

        self.vision_system.process_frames()

        api_request_thread.join()
        face_tracking_thread.join()
        head_movement_thread.join()

if __name__ == "__main__":
    brain = RoboticBrain()
    brain.run()

