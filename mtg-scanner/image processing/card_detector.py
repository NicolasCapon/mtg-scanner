import cv2
import re
import os
import json
import requests
import urllib.request
import cv2
import time
import scipy
from PIL import Image
import imagehash as ih
from time import sleep
from operator import itemgetter
from skimage import exposure
import distance
import numpy as np
import logging
from skimage.measure import compare_ssim as ssim
from langdetect import detect
import pytesseract
from skimage import feature
import database_manager as dbm

class CardDetector:
    """
    Detect name of a card from a photo.
    """
    
    def __init__(self):
        """ImageManager constructor"""
        logging.basicConfig(filename='/home/pi/MTG_scan/logs/cd.log',level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='[%d/%m/%Y %H:%M:%S]')
        self.photos_dir = "/home/pi/MTG_scan/photos"
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.database = dbm.DataBaseManager()
        self.phashes = self.database.get_phashes()
                                
    def resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        """ Resize a photo while saving the ratio, see imutils from https://www.pyimagesearch.com"""
        dim = None
        (h, w) = image.shape[:2]
        
        if width is None and height is None:
            return image
        
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
            
        else:
            r = width  / float(w)
            dim = (width, int(h * r))
            
        resized = cv2.resize(image, dim, interpolation = inter)
        return resized

    def get_framed_card(self, photo):
        """ From a photo, get a 90 top view of the card and crop the background
            come from https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/"""
        ratio = photo.shape[0] / 300.0
        orig = photo.copy()
        photo = self.resize(photo, height = 300)
        
        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        
        # find contours in the edged image, keep only the largest
        # ones, and initialize our screen contour
        (im2, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None
        
        # loop over our contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break
        
        # now that we have our screen contour, we need to determine
        # the top-left, top-right, bottom-right, and bottom-left
        # points so that we can later warp the image -- we'll start
        # by reshaping our contour to be our finals and initializing
        # our output rectangle in top-left, top-right, bottom-right,
        # and bottom-left order
        pts = screenCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype = "float32")
        
        # the top-left point has the smallest sum whereas the
        # bottom-right has the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # compute the difference between the points -- the top-right
        # will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        # multiply the rectangle by the original ratio
        rect *= ratio
        
        # now that we have our rectangle of points, let's compute
        # the width of our new image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        
        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        
        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        
        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        
        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
        
        # convert the warped image to grayscale and then adjust
        # the intensity of the pixels to have minimum and maximum
        # values of 0 and 255, respectively
        
        # warp = exposure.rescale_intensity(warp, out_range = (0, 255))
        # cv2.imwrite(os.path.join(self.photos_dir, "framed_photo.jpg"),warp)

        return warp

    def correlation_coefficient(self, im1, im2):
        """Create a correlation score between two images.
           The higher the score is, the closer images are"""
        # score = scipy.signal.correlate2d(i1, i2)
        # logging.info(score)
        # return score
        
        # product = sum((i1 - np.mean(i1)) * (i2 - np.mean(i2)))
        # stds = ((i1.size - 1) * np.std(i1) * np.std(i2))
        # if stds == 0: return 0
        # else:
            # dist_ncc = product / stds
            # logging.info(dist_ncc)
            # return dist_ncc
            
        product = np.mean((im1 - np.mean(im1)) * (im2 - np.mean(im2)))#.mean()
        stds = np.std(im1) * np.std(im2)#.std()
        if stds == 0:
            return 0
        else:
            product /= stds
            return product
            
    def ccoeff_normed(self, im1, im2):
        tmp1 = np.float32(im1) / 255.0
        tmp2 = np.float32(im2) / 255.0

        cv2.subtract(tmp1, cv2.mean(tmp1), tmp1)
        cv2.subtract(tmp2, cv2.mean(tmp2), tmp2)

        norm1 = tmp1.copy()
        norm2 = tmp1.copy() #cv2.CloneImage(tmp2)
        # cv2.pow(tmp1, norm1, 2.0)
        # cv2.pow(tmp2, norm2, 2.0)

        #cv.Mul(tmp1, tmp2, tmp1)
        # cv2.DotProduct(tmp1, tmp2) / (cv2.sumElems(norm1)[0]*cv2.sumElems(norm2)[0])**0.5
        num = np.dot(tmp1, tmp2.transpose())
        denum = (cv2.sumElems(np.squeeze(np.asarray(norm1)))[0]*cv2.sumElems(np.squeeze(np.asarray(norm2)))[0])**0.5
        return cv2.mean(num / denum)
    
    def get_language(self, photo):
        im = Image.open(photo)
        text = pytesseract.image_to_string(im)
        lang = detect(text)
        return lang
    
    def score_images(self, im1, im2, method="orb"):
        """Score the similarity between two images according to differents methods.
           im1 = framed photo ; im2 = db_im"""
        score = 0
        if method == "ssim":
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            score = ssim(cv2.resize(im1, (im2.shape[1], im2.shape[0]), interpolation = cv2.INTER_AREA), im2, multichannel=False)
        if method == "hist_inter":
            #crop_zone = (20,35,203,171)
            #im1 = im1[20:203, 35:171]
            #im2 = im2[20:203, 35:171]
            # photo_hist = cv2.calcHist([cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0,256])
            # gray_card_im = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            # image_hist = cv2.calcHist([gray_card_im], [0], None, [256], [0,256])
            score = 0
            r = range(0,im1.shape[-1])
            for i in r:
                photo_hist = cv2.calcHist([im1], [i], None, [256], [0,256])
                image_hist = cv2.calcHist([im2], [i], None, [256], [0,256])
                score += cv2.compareHist(photo_hist, image_hist, method = cv2.HISTCMP_INTERSECT)
        if method == "cor":
            im1 = cv2.resize(im1, (im2.shape[1], im2.shape[0]))
            # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            score = cv2.matchTemplate(im2, im1, cv2.TM_CCOEFF_NORMED)
            # score = self.correlation_coefficient(im1, im2)
        if method == "diff":
            im1 = cv2.resize(im1, (im2.shape[1], im2.shape[0]))
            diff = im1 - im2
            matrix = np.array(diff)
            flat = matrix.flatten()
            numchange = np.count_nonzero(flat)
            score = 100 * float(numchange) / float(len(flat))
        if method == "hog":
            im1 = cv2.resize(im1, (im2.shape[1], im2.shape[0]))
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            # H1 = feature.hog(im1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
            # H2 = feature.hog(im2, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
            # score = cv2.compareHist(np.float32(H1), np.float32(H2), method = cv2.HISTCMP_BHATTACHARYYA)
            im1 = np.float32(im1) / 255.0
            im2 = np.float32(im2) / 255.0
            # Calculate gradient 
            im1_gx = cv2.Sobel(im1, cv2.CV_32F, 1, 0, ksize=1)
            im1_gy = cv2.Sobel(im1, cv2.CV_32F, 0, 1, ksize=1)
            im2_gx = cv2.Sobel(im2, cv2.CV_32F, 1, 0, ksize=1)
            im2_gy = cv2.Sobel(im2, cv2.CV_32F, 0, 1, ksize=1)
            # Python Calculate gradient magnitude and direction ( in degrees ) 
            mag1, angle1 = cv2.cartToPolar(im1_gx, im1_gy, angleInDegrees=True)
            mag2, angle2 = cv2.cartToPolar(im2_gx, im2_gy, angleInDegrees=True)
            # Compute corelation between angles
            # (h, w) = angle1.shape[:2]
            # print(h)
            # print(w)
            score = np.nanmin(1 - scipy.spatial.distance.cdist(angle1, angle2, "cosine"))
            # score = self.ccoeff_normed(angle1, angle2)
        if method == "orb":
            # See https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2 = cv2.imread(im2, 0)
            #im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            # Initiate SIFT detector
            orb = cv2.ORB_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = orb.detectAndCompute(im1,None)
            kp2, des2 = orb.detectAndCompute(im2,None)
            logging.info(des1.shape)
            logging.info(des2.shape)
            logging.info(type(des1))
            logging.info(type(des2))
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)
            score = sum(m.distance for m in matches[:20])
            # for m in matches[:20]:
                # score += m.distance
        return score
    
    def orb_score(self, des1, des2):
        """Create a score between two image descriptors"""
        # logging.info(des2)
        # bf.match can be optimized with simplier hamming score calculation
        # logging.info(des1.shape)
        # logging.info(des2.shape)
        # logging.info(type(des1))
        # logging.info(type(des2))
        # matches = sorted(cv2.norm(des1, des2, cv2.NORM_HAMMING))
        # score = sum(m for m in matches[:20])
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        score = sum(m.distance for m in matches[:20])
        return score

    def get_best_match2(self, photo_path):
        """Try to match the photo with the nearest image in the database"""
        #split = cv2.split(warp)
        s = time.time()
        photo = cv2.imread(photo_path)
        # Get the 90° cropped view of the card
        framed = self.get_framed_card(photo)
        photo_array = Image.fromarray(framed)
        # Detect language of the card: https://stackoverflow.com/questions/37235932/python-langdetect-choose-between-one-language-or-the-other-only
        # text = pytesseract.image_to_string(photo_array)
        # try:
            # lang = detect(text)
        # except:
            # lang = "NF"
        phash_photo = str(ih.phash(photo_array))
        # Get 20 best results from phash matching
        distances = sorted([(distance.hamming(phash_photo, i[0]), i[2]) for i in self.phashes], key=itemgetter(0))[:20]
        # Extract descriptor from framed card
        kp1, des_photo = self.orb.detectAndCompute(cv2.cvtColor(framed, cv2.COLOR_BGR2GRAY), None)
        # Get descriptors for the 20 best phash matches
        cards_list = self.database.get_descriptors([p[1] for p in distances])
        # Get the card with the minimum hamming distance between descriptors
        sc, best_match = min([(self.orb_score(des_photo, des_im), path) for des_im, path in cards_list], key = itemgetter(0))
        # scores = []
        # for phash, im_path in distances:
            # scores.append((self.score_images(framed, im_path), im_path))
        # sc, best_match = min(scores, key = itemgetter(0))
        e = time.time()
        logging.info("===> Result : orb_score = {0} / time = {1}s / lang = {2} / best match = {3}".format(sc, e - s, lang, os.path.basename(best_match)))
        print("===> Result : orb_score = {0} / time = {1}s / best match = {2}".format(sc, e - s, os.path.basename(best_match)))
        return best_match

    def get_best_match(self, photo_path):
        """Try to match the photo with the nearest image in the database"""
        #split = cv2.split(warp)
        s = time.time()
        photo = cv2.imread(photo_path)
        # Get the 90° cropped view of the card
        framed = self.get_framed_card(photo)
        photo_array = Image.fromarray(framed)
        # Detect language of the card: https://stackoverflow.com/questions/37235932/python-langdetect-choose-between-one-language-or-the-other-only
        # text = pytesseract.image_to_string(photo_array)
        # try:
            # lang = detect(text)
        # except:
            # lang = "NF"
        phash_photo = str(ih.phash(photo_array))
        # Get 20 best results from phash matching
        distances = sorted([(distance.hamming(phash_photo, i[0]), i[2]) for i in self.phashes], key=itemgetter(0))[:20]
        # Extract descriptor from framed card
        kp1, des_photo = self.orb.detectAndCompute(cv2.cvtColor(framed, cv2.COLOR_BGR2GRAY), None)
        # Get descriptors for the 20 best phash matches
        cards_list = self.database.get_descriptors([p[1] for p in distances])
        # Get the card with the minimum hamming distance between descriptors
        sc, cpath, best_match = min([(self.orb_score(des_photo, des_im), cpath, (name, set_code, scryfall_id)) for des_im, cpath, name, set_code, scryfall_id in cards_list], key = itemgetter(0))
        # scores = []
        # for phash, im_path in distances:
            # scores.append((self.score_images(framed, im_path), im_path))
        # sc, best_match = min(scores, key = itemgetter(0))
        e = time.time()
        logging.info("===> Result : orb_score = {0} / time = {1}s / lang = {2} / best match = {3}".format(sc, e - s, "None", os.path.basename(cpath)))
        print("===> Result : orb_score = {0} / time = {1}s / best match = {2}".format(sc, e - s, os.path.basename(cpath)))
        return best_match 

def get_content(url):
    """Extract data from API json file. If there is multiple pages, gather them.
       See https://scryfall.com/docs/api/lists"""
    if not url: return False
    sleep(0.1)
    r = requests.get(url)
    data = {}
    if r.status_code == requests.codes.ok:
        data = json.loads(r.content.decode('utf-8'))
        if data.get("object", False) == "error": 
            logging.info("API respond an error to url : {0}".format(url))
            return False
        if data.get("has_more", None) and data.get("next_page", None):
            content = get_content(data["next_page"])
            data["data"] += content.get("data", [])
    return data  
    
# cd = CardDetector()
# cd.get_best_match("/home/pi/telegram/GeekStreamBot/images/card_to_detect.jpg")
# print("END")

# logging.info("========== START NEW DETECTION ==========")
# for file in os.listdir(cd.photos_dir):
    # #See Watchdog for folder monitoring
    # logging.info("________________{0}________________".format(file))
    # photo_path = os.path.join(cd.photos_dir, file)
    # cd.get_best_match(photo_path)
    
# logging.info("END")