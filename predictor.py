# organize imports
from __future__ import absolute_import, division, print_function

import imutils
import numpy as np
from PIL import Image
import cv2
from tensorflow import keras

# global variables
bg = None


def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def main():
    new_model = keras.models.load_model('my_model.h5')
    class_names = ['one', 'two', 'three', 'four', 'five']

    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    image_num = 0

    start_recording = False
    # sample_name = 'two'

    # keep looping, until interrupted
    while (True):
        # get the current frame
        (grabbed, frame) = camera.read()
        if (grabbed == True):

            # resize the frame
            frame = imutils.resize(frame, width=700)

            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()

            # get the height and width of the frame
            (height, width) = frame.shape[:2]

            # get the ROI
            roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our running average model gets calibrated
            if num_frames < 30:
                run_avg(gray, aWeight)
                print(num_frames)
            else:
                # segment the hand region
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand
                    cv2.imwrite('predict.png', thresholded)
                    basewidth = 100
                    img = Image.open('predict.png')
                    wpercent = (basewidth / float(img.size[0]))
                    hsize = int((float(img.size[1]) * float(wpercent)))
                    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
                    img.save('predict.png')
                    image = cv2.imread('predict.png')
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray_image = gray_image / 255
                    gray_image = np.reshape(gray_image, [1, 89, 100])
                    prediction = new_model.predict(gray_image)
                    print(class_names[int(np.argmax(prediction))])
                    cv2.putText(clone, str(class_names[int(np.argmax(prediction))]), (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),thickness=2)

                    # draw the segmented region and display the frame
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                    cv2.imshow("Thesholded", thresholded)

            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            # increment the number of frames
            num_frames += 1

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord("q") or image_num > 100:
                break

            if keypress == ord("s"):
                start_recording = True

        else:
            print("[Warning!] Error input, Please check your(camra Or video)")
            break

    # free up memory
    camera.release()
    cv2.destroyAllWindows()


main()
