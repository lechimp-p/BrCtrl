from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import numpy as np
import pytesseract
import threading

HOST = "localhost"
PORT = 8080
DIGITS_ON_METER = 8.0
SIGNIFICANT_DIGITS_ON_METER = 5.0
MAX_METER_STEP = 10
VIDEO_INPUT = 0
METRIC_NAME = "gas_volume_in"

def readCurrentMeterStatus(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 90, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    _, w_total = image .shape[:2]

    for c in contours:
        approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
        if len(approx) == 4:
            _,_,w,_ = cv2.boundingRect(c)
            if w/w_total > 0.3:
                candidates.append(c)
                image = cv2.drawContours(image, [c], -1, (0,0,0), 10)

    if len(candidates) != 1:
        raise RuntimeError("Found more then one candidate regions.")

    x,y,w,h = cv2.boundingRect(candidates[0])
    w = int(w / DIGITS_ON_METER * SIGNIFICANT_DIGITS_ON_METER)
    orig = np.float32([[x,y],[x,y+h],[x+w,y+h], [x+w,y]])
    targ = np.float32([[0,0],[0,h],[w,h],[w,0]])
    trafo = cv2.getPerspectiveTransform(orig, targ)
    image = cv2.warpPerspective(img, trafo, (w,h), flags=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, 0)
    inv = cv2.bitwise_not(thresh)

    return pytesseract.image_to_string(inv, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789').strip()

def getNext(current, proposed):
    if len(proposed) != SIGNIFICANT_DIGITS_ON_METER:
        return current

    proposed = int(proposed)
    diff = proposed - current 
    if diff < 0:
        return current

    if diff > MAX_METER_STEP:
        return current

    return proposed

METER_STATE = 4710 

class MeterMetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(bytes("%s %s" % (METRIC_NAME, METER_STATE), "utf-8"))

READ_METER = True

def read_meter(cam):
    global METER_STATE
    while READ_METER:
        try:
            _, frame = cam.read()
            METER_STATE = getNext(METER_STATE, readCurrentMeterStatus(frame)) 
            print("Meter State: %s" % (METER_STATE))
        except Exception as e:
            print(e)

if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), MeterMetricsHandler)
    print("Start server at http://%s:%s." % (HOST, PORT))

    cam = cv2.VideoCapture(VIDEO_INPUT)
    if not cam.isOpened():
       raise RuntimeError("Cannot open video input") 

    reader_thread = threading.Thread(target=read_meter, args=(cam,))
    reader_thread.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
       pass

    READ_METER = False
    reader_thread.join()    

    cam.release()
    server.server_close()
    print("Server stopped.") 
