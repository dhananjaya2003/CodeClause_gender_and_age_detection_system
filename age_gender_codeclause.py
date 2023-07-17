# Library Installation
import cv2

# faceBox - Detecting face with it's standard parameter
def faceBox(faceNet,frame):
  print(frame)
  frameWidth  = frame.shape[1]
  frameHight = frame.shape[0]
  blob = cv2.dnn.blobFromImage(frame,1.0, (227,227), [104,117,123],swapRB=False)
  faceNet.setInput(blob)
  detection = faceNet.forward()
  bboxs = []
  for i in range(detection.shape[2]):
    confidence = detection[0,0,i,2]
    if confidence > 0.7:
      x1 = int(detection[0,0,i,3]*frameWidth)
      y1 = int(detection[0,0,i,4]*frameHight)
      x2 = int(detection[0,0,i,5]*frameWidth)
      y2 = int(detection[0,0,i,6]*frameHight)
      bboxs.append([x1,y1,x2,y2])

      cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
  return frame, bboxs

# Pre-defined Models
faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Reading the Models in cv2
faceNet = cv2.dnn.readNet(faceModel,faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

# Standard values taken from google for Model
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Capturing Video for Detection
video = cv2.VideoCapture(0)

# Determining Gender and Age
while True:
  ret,frame = video.read()
  frame,bboxs = faceBox(faceNet,frame)
  for bbox in bboxs:
    face = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    blob = cv2.dnn.blobFromImage(face, 1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)
    genderNet.setInput(blob)
    genderPred = genderNet.forward()
    gender = genderList[genderPred[0].argmax()]

    ageNet.setInput(blob)
    agePred = ageNet.forward()
    age = ageList[agePred[0].argmax()]

    label = "{},{}".format(gender,age)
    cv2.putText(frame,label,(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
  cv2.imshow('Age-Gender',frame)
  k = cv2.waitKey(1)

  if k == ord('q'):
    break

#video.release()
cv2.destroyAllWindows()
