import os
import cv2
import sys
import time
from PIL import Image
import multiprocessing
import card_detector
import imagehash as ih
from operator import itemgetter
import distance
import numpy as np

def multi_run(params):
    p = multiprocessing.Pool()
    results = p.imap_unordered(orb_matching, params)
    p.close()
    p.join()
    # try:
        # print("get")
        # res = results.get(timeout=3)
    # except multiprocessing.TimeoutError:
        # print("error")
        # p.terminate()
        # p.join()
        # res=[]
    # print(res)
    return results
    
def orb_matching(params):
    """ orb = cv2.ORB_create() // im1 = photo_array // im2 = imdb"""
    # See https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    s = time.time()
    des1, des2 = params
    # im2 = cv2.imread(im_db, 0)
    # orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    # kp1, des1 = orb.detectAndCompute(photo,None)
    # kp2, des2 = orb.detectAndCompute(im2, None)
    # create BFMatcher object

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Try to calcul hamming distance between descriptors:
    # dist = cv2.norm(descriptor1, descriptor2, cv2.NORM_HAMMING)
    matches = sorted(matches, key = lambda x:x.distance)
    score = 0
    for m in matches[:20]:
        score += m.distance

    return time.time() - s

if __name__ == '__main__':    
    cd = card_detector.CardDetector()
    print("========== START NEW DETECTION ==========")
    for file in os.listdir(cd.photos_dir):
        
        print("________________{0}________________".format(file))
        photo_path = os.path.join(cd.photos_dir, file)
        photo = cv2.imread(photo_path)
        framed = cd.get_framed_card(photo)
        photo_array = Image.fromarray(framed)
        phash_photo = str(ih.phash(photo_array))
        distances = sorted([(distance.hamming(phash_photo, i[0]), i[2]) for i in cd.phashes], key=itemgetter(0))[:20]
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(cv2.cvtColor(framed, cv2.COLOR_BGR2GRAY), None)
        
        params = []
        for im_phash, scryfall_id, image_path in distances:
            im2 = cv2.imread(image_path, 0)
            s = time.time()
            kp2, des2 = orb.detectAndCompute(im2, None)
            e = time.time()
            print("detectAndCompute time = {0}s".format(e - s))
            params.append((des1,des2))
        
        results = []
        for param in params:
            print("orb_matching time = {0}s".format(orb_matching(param)))
        # results = multi_run(params)
        # cd.get_best_match(photo_path)
        # lang = cd.get_language(photo_path)
        # print("===> Result : Language = {0}".format(lang))