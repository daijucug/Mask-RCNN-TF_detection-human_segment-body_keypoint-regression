import cv2
import struct
import socket
import numpy as np



def sendImageThroughSocket(socket,image):
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    imageSize = len(img_str)
    val = struct.pack('!i', imageSize)
    socket.send(val)
    socket.send(img_str)
    print imageSize

def getImageFromSocket(s):
    imageSize = s.recv(4)
    imageSize = imageSize[::-1]
    imageSize = struct.unpack('i', imageSize)[0]
    imageBytes = b''
    while imageSize > 0:
        chunk = s.recv(imageSize)
        imageBytes += chunk
        imageSize -= len(chunk)

    data = np.fromstring(imageBytes, dtype='uint8')
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image

video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture('/home/alex/PycharmProjects/youtube/fail.mkv')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/alex/PycharmProjects/youtube/output3.avi',fourcc, 20.0, (640,360))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect(('10.12.5.208', 5555))
s.connect(('10.20.5.205', 5555))
i =0
while True:
    ret,image = video_capture.read()
    # i = i+1
    # if i%5 != 0:
    #     ret,image = video_capture.read()
    if ret==True:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        h,w = image.shape[0],image.shape[1]
        ratio = float(h)/float(w)
        if h >=w:
            if h>640:
                h=640
                w=int(h/ratio)
        elif w>h:
            if w >640:
                w=640
                h=int(ratio*w)
        image = cv2.resize(image,(w,h))
        print w,h

        sendImageThroughSocket(s,image)
        image = getImageFromSocket(s)
        cv2.imshow("image",image)
        cv2.waitKey(20)

        out.write(image)
    else:
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()
