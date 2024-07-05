import cv2
import numpy as np
import cv2 as cv
import glob

# termination criteria
chessboard = (7,7)
imageSize = (2608,4624)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard[0]*chessboard[1],3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard[1], 0:chessboard[0]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('ImageCalibration3D\*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboard, None)
    cv.namedWindow("img", cv2.WINDOW_NORMAL)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboard, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imageSize, None, None)
np.savetxt('Datasets\\CameraMatrix.txt', mtx, fmt='%.8e')
print("camera matrix: \n", mtx)
