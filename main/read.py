import cv2 as cv
import os 
import numpy as np


def open_file (path):
    full_path = os.path.join(os.path.dirname(__file__), f'../../{path}')
    assert os.path.exists(full_path)
    return full_path

def main():
    path = open_file("visual-neglect/photos/detergent_cabniet.jpg")
    video = open_file("visual-neglect/videos/walkinggroceryaisle.mp4")
    
    walking = cv.VideoCapture(video)
    

    
    while True:
        isTrue, frame = walking.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray,(5,5),0)
        ret, thresh_img = cv.threshold(blur,91,255,cv.THRESH_BINARY)
        contours =  cv.findContours(thresh_img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[-2]
        for c in contours:
            cv.drawContours(frame, [c], -1, (0,0,255), 1)

        # make a rectangle
        cv.rectangle(frame,(0,0),(20,1000),(0,0,255), -1)
        # Display the resulting frame
        cv.imshow('Walking Down an Aisel', frame)

        if cv.waitKey(20) & 0xFF==ord('d'):
            break
    
    walking.release()
    
    # cv.imshow("Finding a Detergent in a Cabniet", detergent)
    # cv.waitKey(10) 
    # cv.destroyAllWindows()


if __name__ == "__main__":
    main()
