# coding:utf-8

import os
import cv2
import numpy as np
from time import sleep
import time


def init_feature():
    detector = cv2.xfeatures2d.SURF_create(500)  # 500 is the threshold Hessian value for the detector.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)  # Or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return detector, matcher


def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, img1.shape[2]), np.uint8)
    vis[:h1, :w1, :] = img1
    vis[:h2, w1:w1 + w2, :] = img2

    if len(kp_pairs) is 0:
        return vis

    if H is not None and len(status) > 10:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (0, 0, 255), thickness=2)

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)

    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
            cv2.line(vis, (x1, y1), (x2, y2), green)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)

    cv2.imshow(win, vis)
    return vis


def filter_matches(kp1, kp2, matches, ratio=0.75):
    good_matches = [m[0] for m in matches if m[0].distance <= m[1].distance * ratio]

    kp_pairs = [(kp1[m.queryIdx], kp2[m.trainIdx]) for m in good_matches]
    p1 = np.float32([kp[0].pt for kp in kp_pairs])
    p2 = np.float32([kp[1].pt for kp in kp_pairs])
    return p1, p2, kp_pairs


if __name__ == '__main__':
    path_name = './img2/'
    winName = 'Detector'
    file_name, object_name = './img/1.jpeg', 'A'
    img1 = cv2.imread(file_name)
    height, width, _ = img1.shape
    print(width, height)
    img_height = 500
    img_width = width * img_height // height
    print(img_width, img_height)
    img1 = cv2.resize(src=img1, dsize=(img_width, img_height))
    detector, matcher = init_feature()
    kp1, desc1 = detector.detectAndCompute(img1, None)

    cap = cv2.VideoCapture("./test.m4v")
    cv2.namedWindow(winName)

    first_pic = cv2.imread("0.jpeg")

    print('Start..')

    match_dic = {}
    for pic_name in os.listdir(path_name)[1:]:
        pic_name = path_name + pic_name
        if os.path.isdir(pic_name):
            continue
        img_pic = cv2.imread(pic_name)
        height, width, _ = img_pic.shape
        img_width = width * img_height // height
        img_pic = cv2.resize(src=img_pic, dsize=(img_width, img_height))
        kp_pic, desc_pic = detector.detectAndCompute(img_pic, None)
        match_dic[pic_name] = (kp_pic, desc_pic)

    while True:

        time_b = time.time()

        sleep(0.05)
        s, img_frame = cap.read()
        frame_height = img_height
        frame_width = 640 * frame_height // 480
        img_frame = cv2.resize(img_frame, (frame_width, frame_height))

        kp_frame, desc_frame = detector.detectAndCompute(img_frame, None)

        if desc_frame is None:
            continue
        max_match = 5
        match_or_not = -1
        max_match_pic = ''


        for pic_name in os.listdir(path_name)[1:]:
            pic_name = path_name + pic_name
            if os.path.isdir(pic_name):
                continue
            kp_pic, desc_pic = match_dic[pic_name]
            raw_matches = matcher.knnMatch(desc_pic, trainDescriptors=desc_frame, k=2)
            p1, p2, kp_pairs = filter_matches(kp_pic, kp_frame, raw_matches, 0.5)
            if len(p1) > max_match:
                match_or_not = 1
                max_match = len(p1)
                max_match_pic = pic_name

        time_e = time.time()

        print(time_e - time_b)

        if match_or_not > 0:
            img_pic = cv2.imread(max_match_pic)
            first_pic = img_pic
            height, width, _ = img_pic.shape
            img_width = width * img_height // height
            img_pic = cv2.resize(src=img_pic, dsize=(img_width, img_height))
            kp_pic, desc_pic = detector.detectAndCompute(img_pic, None)
            raw_matches = matcher.knnMatch(desc_pic, trainDescriptors=desc_frame, k=2)
            p1, p2, kp_pairs = filter_matches(kp_pic, kp_frame, raw_matches, 0.5)
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            vis = explore_match(winName, img_pic, img_frame, kp_pairs, status, H)
        else:
            show_pic = first_pic
            height, width, _ = show_pic.shape
            img_width = width * img_height // height
            show_pic = cv2.resize(src=show_pic, dsize=(img_width, img_height))
            h1, w1 = show_pic.shape[:2]
            h2, w2 = img_frame.shape[:2]
            vis = np.zeros((max(h1, h2), w1 + w2, show_pic.shape[2]), np.uint8)
            vis[:h1, :w1, :] = show_pic
            vis[:h2, w1:w1 + w2, :] = img_frame
            cv2.imshow(winName, vis)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
