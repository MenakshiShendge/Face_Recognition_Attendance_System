import cv2
import openpyxl
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import os

excel_file_path = 'attendance.xlsx'

if os.path.exists(excel_file_path):
    wb = openpyxl.load_workbook(excel_file_path)
    ws = wb.active
else:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['ID', 'Name', 'Timestamp'])

known_faces = {}
known_names = {}
attendance_record = set()

captured_faces_dir = 'captured_faces'
if not os.path.exists(captured_faces_dir):
    os.makedirs(captured_faces_dir)

def mark_attendance(student_id, name):
    if student_id not in attendance_record:
        ws.append([student_id, name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        wb.save(excel_file_path)
        attendance_record.add(student_id)

def capture_face(student_id, name):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            try:
                face_resized = cv2.resize(face_roi, (100, 100))
            except cv2.error as e:
                print(f"Error resizing face: {e}")
                continue

            cv2.imshow('Captured Face', face_resized)


            face_filename = f'{captured_faces_dir}/student_{student_id}.png'
            cv2.imwrite(face_filename, face_resized)


            key = cv2.waitKey(1)
            if key == 32:  # ASCII code for space key
                known_faces[student_id] = face_filename
                known_names[student_id] = name
                cv2.destroyAllWindows()
                return
            elif key == 27:  # ASCII code for escape key
                cv2.destroyAllWindows()
                return

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def load_known_faces(csv_file_path):
    with open(csv_file_path, 'r') as file:
        for line in file:
            student_id, name, *_ = line.strip().split(',')
            known_names[student_id] = name


csv_file_path = 'known_faces.csv'  # Change this to your CSV file path
load_known_faces(csv_file_path)

student_id = input("Enter Student ID: ")
name = input("Enter Student Name: ")


capture_face(student_id, name)

# Open the webcam for attendance marking
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through each face in the current frame
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face_roi = gray[y:y+h, x:x+w]

        # Resize the face to a fixed size
        face_resized = cv2.resize(face_roi, (100, 100))

        # Compare with known faces
        for known_id, known_face_path in known_faces.items():
            # Load the known face
            try:
                known_face = cv2.imread(known_face_path, cv2.IMREAD_GRAYSCALE)
                known_face_resized = cv2.resize(known_face, (100, 100))
            except cv2.error as e:
                print(f"Error loading known face: {e}")
                continue

            # Compute Structural Similarity Index (SSI)
            similarity_index, _ = ssim(known_face_resized, face_resized, full=True)

            # Set a similarity threshold (adjust as needed)
            if similarity_index > 0.7:
                # Draw a rectangle around the recognized face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Mark attendance
                mark_attendance(known_id, known_names[known_id])

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
