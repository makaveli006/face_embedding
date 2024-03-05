import cv2
import os
import time
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
from scipy.spatial.distance import cosine
from keras_facenet import FaceNet
import tensorflow as tf

def detect_face_roi(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    largest_face = None
    closest_face = None
    min_distance = float('inf')
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        distance = ((center[0] - image.shape[1]//2)**2 + (center[1] - image.shape[0]//2)**2)**0.5
        if w * h > (largest_face[2] * largest_face[3] if largest_face is not None else 0):
            largest_face = (x, y, w, h)
        if distance < min_distance:
            closest_face = (x, y, w, h)
            min_distance = distance
    (x, y, w, h) = largest_face or closest_face
    face_roi = image[y:y+h, x:x+w]

    return face_roi

def FrameCapture(video_path,output_path):
    for file in os.listdir(output_path):
        os.remove(f'{output_path}/{file}')
    vidObj = cv2.VideoCapture(video_path)
  
    count = 0
  
    success = 1
  
    while success:
        success, image = vidObj.read()
        try:
            face_roi = detect_face_roi(image)
            face_roi_resized = cv2.resize(face_roi,(256,256))
        except:
            pass
        cv2.imwrite(f"{output_path}/frame%d.jpg" % count, face_roi_resized)
  
        count += 1
def SaveFrame(video_path,output_path):

    try:
        FrameCapture(video_path,output_path)
    except:
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if(length == len(os.listdir(output_path))):
            pass
        else:
            print('An Error occured during extracting registration_frames.')
            print(f'[registration_frames].....{os.listdir(output_path)}registration_frames saved.')

def extract_face_embeddings(image_path, model):

    try:
        img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
        img = tf.keras.utils.img_to_array(img)
    except:
        img = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    img = np.expand_dims(img, axis=0)
    img = (img - 127.5) / 128.0
    
    embeddings = model.model.predict(img)
    return embeddings[0]

def face_similarity_of_two_embeddings(emp1,emp2):

    distance = cosine(emp1,emp2)

    return distance

def get_face_embeddings_(frame_folder,model):
    face_embeddings = []
    for i in os.listdir(f'{frame_folder}/'):
        single_face_embedding = extract_face_embeddings(f'{frame_folder}/{i}',model=model)
        face_embeddings.append(single_face_embedding)
    return face_embeddings

def register_newuser(video_path,output_path,model):
    SaveFrame(video_path=video_path,output_path=output_path)
    face_embeddings = get_face_embeddings_(frame_folder=output_path,model=model)
    return face_embeddings

def authenticate_user(video_path,model):
    SaveFrame(video_path=video_path,output_path='Images/authentication_frames/')
    face_embeddings = get_face_embeddings_(frame_folder='Images/authentication_frames/',model=model)
    return face_embeddings

def is_user_valid(face_embeddings,auth_embeddings):
    total_score = []
    for i in face_embeddings:
        for j in auth_embeddings:
            similarity = face_similarity_of_two_embeddings(i,j)
            total_score.append(similarity)
    total_score_avg = sum(total_score)/len(total_score)
    if(total_score_avg<.3):
        return True
    else:
        return False

def record_webcame(frame_count = 600,output_path='videos/',model=FaceNet(),registered_emb = []): 
    user_valid = 0
    embedding_list = [] 
    cap = cv2.VideoCapture(0)
    i = 0
    flag = True
    while flag:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            try:
                frame = detect_face_roi(frame)
                frame = cv2.resize(frame,(256,256))
                emb = extract_face_embeddings(frame,model)
                embedding_list.append(emb)
                i+=1
                if(i%15==0):
                    if(is_user_valid(registered_emb,embedding_list)):
                        user_valid = 1
                        flag = False
                        break
                    else:
                        embedding_list = []
                elif (i>=frame_count):
                    flag = False
                    break
            except:
                pass
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    if(user_valid==1):
        return True
    else:
        return False

def main(new_reistration_video_path,registerd_user_auth_face_embeddings,key):

    model = FaceNet()
    if(key=='user_registration'):
        registerd_user_face_embeddings = register_newuser(video_path=new_reistration_video_path,output_path='images/registration_frames/',model=model)

        return registerd_user_face_embeddings
    elif(key=='user_authentication'):
        return record_webcame(registered_emb=registerd_user_auth_face_embeddings)

if __name__ == '__main__':
    main()