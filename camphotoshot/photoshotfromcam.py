import cv2
import os

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a folder to store the images
folder_path = "/home/nammon/Desktop/my_workspace/workspace/CS64 Project/dataset_img_test/64020823"

# ตรวจสอบว่าโฟลเดอร์มีอยู่แล้วหรือไม่ ถ้าไม่มีให้สร้างขึ้นมา
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Initialize the image counter
img_counter = 0

while img_counter < 50:
    # Capture frame-by-frame
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If a face is detected, save the image
    if len(faces) > 0:
        img_name = f"{folder_path}/face_{img_counter+1}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Image {img_counter+1} saved!")
        img_counter += 1
    
    # Display the frame
    cv2.imshow("Webcam", frame)
    
    # Check for 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cam.release()
cv2.destroyAllWindows()

print(f"Captured {img_counter} images. Program finished.")