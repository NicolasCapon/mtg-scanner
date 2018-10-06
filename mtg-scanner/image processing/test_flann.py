import cv2
import json
import sqlite3
import numpy as np
import card_detector as cd
import datetime
from collections import Counter
import logging

# https://stackoverflow.com/questions/46479237/find-image-from-a-database-of-images
# TODO : Tester avec moins de points que 500 (50 minimum)
logging.basicConfig(filename='/home/pi/MTG_scan/logs/flann.log',level=logging.INFO, format='%(asctime)s %(message)s', datefmt='[%m/%d/%Y %H:%M:%S]')

def get_match(flann, des):
    # descriptors = get_descriptors("c17")
    # des_all = None
    # indexList = []
    # for index, (descriptor, name) in enumerate(descriptors):
        # indexList.extend([index]*descriptor.shape[0])
        # if des_all is None:
            # des_all = descriptor
        # else:
            # des_all = np.concatenate((des_all, descriptor))
    
    indexes, matches = flann.knnSearch(des, 1)
    indexes = [indexList[index[0]] for index in indexes]
    
    likelyMatch_index = max(set(indexes), key=indexes.count)
    likelyMatch_des, likelyMatch_name = descriptors[likelyMatch_index]
    
    return likelyMatch_name

def get_descriptors(set_code):
    """ Return a list of tuple (descriptors, path) from a list of path"""
    conn = sqlite3.connect("/home/pi/MTG_scan/database/MTG.db")
    c = conn.cursor()
    # path_list can only be a list of string to apply join
    req = "SELECT descriptor, name FROM cards WHERE set_code = ?"
    try:
        descriptors = c.execute(req, (set_code, )).fetchall()
    except sqlite3.Error as e:
        print("An SQL error [{0}] occurred with following req : {1}".format(e, req))
        return False
    # Transform string from database to numpy array
    mylist = []
    total = len(descriptors)
    for index, (d, n) in enumerate(descriptors):
        mylist.append((np.array(json.loads(d), dtype=np.uint8), n))
        logging.info("[{}/{}]".format(index, total))
    return mylist # [(np.array(json.loads(d), dtype=np.uint8), n) for d, n in descriptors]

#____________________________________________________________________________________________
def build_index(des_all=None):
    if des_all:
        flann = cv2.flann.Index()
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
        logging.info("Training...")
        flann.build(des_all, index_params)
    else:
        with open("/mnt/PIHDD/raspberry/flann/descriptors.txt", "r") as file:
            des_all = np.array(json.load(file), dtype=np.uint8)
        flann = cv2.flann.Index()
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
        logging.info("Training...")
        flann.build(des_all, index_params)
    return flann, des_all
    
def create_index():
    conn = sqlite3.connect("/home/pi/MTG_scan/database/MTG.db")
    cursor = conn.cursor()
    req = "SELECT descriptor FROM cards"
    cursor.execute(req)

    indexList, des_all = [], None
    for index, row in enumerate(cursor):
        logging.info("[{}]".format(index))
        descriptor = np.array(json.loads(row[0]), dtype=np.uint8)
        indexList.extend([index]*descriptor.shape[0])
        if des_all is None:
            des_all = descriptor
        else:
            des_all = np.concatenate((des_all, descriptor))

    with open("/mnt/PIHDD/raspberry/flann/descriptors.txt", "w") as f:
        f.write(json.dumps(des_all.tolist()))

    with open("/mnt/PIHDD/raspberry/flann/indexes.txt", "w") as f:
        f.write(json.dumps(indexList))
        
    flann, des_all = build_index(des_all)
    logging.info(datetime.datetime.now())
    logging.info("Saving...")
    flann.save("/mnt/PIHDD/raspberry/flann/index")

def find_card(image_path):

    logging.info("Loading image...")
    orb = cv2.ORB_create()
    img = cv2.imread(image_path, 0)
    kp, des = orb.detectAndCompute(img, None)
    
    logging.info("Loading indexes...")
    with open("/mnt/PIHDD/raspberry/flann/indexes.txt", "r") as file:
        indexList = np.array(json.load(file), dtype=np.float32) #np.uint8)
    
    logging.info("Loading descriptors...")
    # flann, des_all = build_index()
    flann = cv2.flann.Index()
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
    flann.load(np.empty((17351998, 0), dtype=np.float32), "/mnt/PIHDD/raspberry/flann/index")
        
    logging.info("Matching...")
    match_ind, matches = flann.knnSearch(des, 1, params={})
    indexes = [indexList[index[0]] for index in match_ind]
    likelyMatch_index = max(set(indexes), key=indexes.count)
    likelyMatch_des = des_all[likelyMatch_index]
    logging.info(likelyMatch_des)
    
find_card('/home/pi/MTG_scan/photos/Bladewing the Risen.jpg')
# logging.info("Matching...")
# logging.info(datetime.datetime.now())
# indexes, matches = flann.knnSearch(des, 1, params={})
# logging.info(datetime.datetime.now())
# indexes = [indexList[index[0]] for index in indexes]

# likelyMatch_index = max(set(indexes), key=indexes.count)
# likelyMatch_des, likelyMatch_name = descriptors[likelyMatch_index]
# logging.info(likelyMatch_name)
# logging.info("End of Matching.")

#____________________________________________________________________________________________
# logging.info(datetime.datetime.now())
# des_all = None
# flann = cv2.flann.Index() # print(dir(flann)) # FLANN_INDEX_KDTREE = 1
# FLANN_INDEX_LSH = 6 # cv2.NORM_HAMMING
# index_params = dict(algorithm=FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)

# descriptors = get_descriptors("c17")
# indexList = []
# for index, (descriptor, name) in enumerate(descriptors):
    # indexList.extend([index]*descriptor.shape[0])
    # if des_all is None:
        # des_all = descriptor
    # else:
        # des_all = np.concatenate((des_all, descriptor))

# with open("/home/pi/MTG_scan/logs/des.txt", "w") as f:
    # f.write(json.dumps(des_all.tolist()))

# with open("/home/pi/MTG_scan/logs/indexes.txt", "w") as f:
    # f.write(json.dumps(indexList))

# logging.info("Training...")
# flann.build(des_all, index_params)

# orb = cv2.ORB_create()
# img = cv2.imread('/home/pi/MTG_scan/photos/Bladewing the Risen.jpg', 0)
# kp, des = orb.detectAndCompute(img, None)

# logging.info("Matching...")
# indexes, matches = flann.knnSearch(des, 1, params={})
# indexes = [indexList[index[0]] for index in indexes]

# likelyMatch_index = max(set(indexes), key=indexes.count)
# likelyMatch_des, likelyMatch_name = descriptors[likelyMatch_index]
# logging.info(likelyMatch_name)
# logging.info("End of Matching.")
# logging.info(datetime.datetime.now())

# print("Saving...")
# flann.save("/home/pi/MTG_scan/database/index")
# print("Loading...")
# flann2 = cv2.flann.Index()
# flann2.load(des_all, "/home/pi/MTG_scan/database/index")
# print(get_match(flann2, des))
