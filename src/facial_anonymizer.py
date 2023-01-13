import time
import os 
import subprocess
import sys  
try:
    import cv2 as cv 
except ImportError:
    sys.stderr.write("OpenCV required for program execution")
try:
    import Mediapipe as mp 
except ImportError:
    sys.stderr.write("Mediapipe required for progam execution")


class FaceDetector():
    def __init__(self, minDetectionConf = 0.75):
        self.minDetectionConf = minDetectionConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConf)
    
    def anonFaces(self, frame):
        ih, iw, ic = frame.shape
        self.results = self.faceDetection.proccess(frame)
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                boundingboxdata = detection.location_data.relative_bounding_box
                thebox = int(boundingboxdata.xmin*iw) - 10, int(boundingboxdata.ymin*ih) - 10, int(boundingboxdata.width*iw*1.2), int(boundingboxdata.height*ih*1.2)
                cv.putText(frame, f"{int(detection.score[0]*100)}%", (int(boundingboxdata.xmin*iw), int(boundingboxdata.ymin*ih) - 40), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)
                cv.rectangle(frame, thebox, (0,0,0), thickness=-1)
        
        return frame




def main():
    try:
        subprocess.run("touch programlog.txt")
    except:
        sys.stderr.write("Cannot create log file")
        sys.exit()
    #opening and writing log file
    logfile = open("programlog.txt", "w")

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        logfile.write("cannot open video source")
        logfile.close()
        sys.stderr.write("cannot open video source")
        sys.exit()
    pTime = 0
    detector = FaceDetector()
    while True:
        isTrue, frame = cap.read()
        if not isTrue:
            logfile.write("cannot recieve frames")
            logfile.close()
            break
        frame = detector.anonFaces(frame=frame)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv.putText(frame, f"FPS: {int(fps)}", (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv.imshow("video", frame)
        if cv.waitKey(1) == ord('d'):
            break
    cap.releast()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()