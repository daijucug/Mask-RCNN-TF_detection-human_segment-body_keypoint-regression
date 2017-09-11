import cv2
import struct
import socket
import numpy as np
import cv2

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

def sendImageThroughSocket(socket,image):
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    imageSize = len(img_str)
    val = struct.pack('!i', imageSize)
    socket.send(val)
    socket.send(img_str)
    print imageSize

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('10.20.5.205', 5555))
serversocket.listen(5)

(clientsocket, address) = serversocket.accept()
image = getImageFromSocket(clientsocket)
#cv2.imshow("image",image)
#cv2.waitKey(1000)
sendImageThroughSocket(clientsocket,image)


