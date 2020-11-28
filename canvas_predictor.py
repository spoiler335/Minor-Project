import cv2
import numpy as np
from tensorflow.keras.models import load_model


model=load_model('mnistModel.h5',compile = False)

img=np.ones([300,300],dtype='uint8')*255;
window_name='Canvas Digit Predictor'
cv2.namedWindow(window_name)
status=False

print("Press p for Predicting the Digit")
print("Press c for clearing the screen")
print("Press q  to quit")

def dig(event,x,y,flag,param):
    global status
    if(event==cv2.EVENT_LBUTTONDOWN):
        status=True
    elif(event==cv2.EVENT_MOUSEMOVE):
        if(status==True):
            cv2.circle(img,(x,y),5,(0,0,0),-2)
    elif(event==cv2.EVENT_LBUTTONUP):
        status=False

cv2.setMouseCallback(window_name,dig)

while(True):
    cv2.imshow(window_name,img)
    key=cv2.waitKey(1)
    if(key==ord('q')):
        break
    elif(key==ord('p')):
        number=img[50:300,50:300]
        cv2.imshow("Corpped",number)
        number=cv2.resize(number,(28,28)).reshape(1,28,28)
        print(np.argmax(model.predict(number)))
    
    elif(key==ord('c')):
        img[:,:]=255
        

cv2.destroyAllWindows()