import cv2

def detect_face_roi(image):
    # Load the pre-trained Haar Cascade face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load the image
    image = cv2.imread(image)
    
    # Check if the image is loaded successfully
    if image is None:
        print("Failed to load the image. Please check if the image path is correct.")
        return None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Find the largest face or the face closest to the center
    largest_face = None
    closest_face = None
    min_distance = float('inf')
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        distance = ((center[0] - image.shape[1]//2)**2 + (center[1] - image.shape[0]//2)**2)**0.5
        if largest_face is None or w * h > largest_face[2] * largest_face[3]:
            largest_face = (x, y, w, h)
        if distance < min_distance:
            closest_face = (x, y, w, h)
            min_distance = distance

    # Extract the largest or closest face
    (x, y, w, h) = largest_face or closest_face
    face_roi = image[y:y+h, x:x+w]
    return face_roi