# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:54:03 2017

@author: ncapon
"""
import re
import os
import json
import requests
import urllib.request
import cv2
import time
from PIL import Image
import imagehash as ih
from time import sleep
from operator import itemgetter
from skimage import exposure
import distance
import numpy as np
from skimage.measure import compare_ssim as ssim
from bs4 import BeautifulSoup
import unidecode
 
class ImageManager: 
    """
    Classe de gestion de la base des images (basé sur Scryfall.com)
    - https://scryfall.com/docs/api
    """
    
    def __init__(self, directory):
        """Constructeur de l'ImageManager"""
        self.directory = directory
        self.forbidden_chars = r'[\\/*?:"<>|]'
        self.forbidden_codes = ["con","nul","prn", "aux","com1","com2","com3","com4",
                                "com5","com6","com7","com8","com9","lpt1","lpt2","lpt3",
                                "lpt4","lpt5","lpt6","lpt7","lpt8","lpt9"]
        
    def get_set_list(self):
        """Récupération de la liste des sets Magic"""
        url = "https://api.scryfall.com/sets"
        content = self.get_content(url)
        return content.get("data", None)
    
    def get_folders_list(self):
        """Méthode qui retourne la liste des folders (un folder par édition)"""
        return [f for f in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, f))]
    
    def get_content(self, url):
        """Méthode qui extrait les données pour une url en tenant compte de la pagination"""
        if not url: return False
        sleep(0.1)
        r = requests.get(url)
        data = {}
        if r.status_code == requests.codes.ok:
            data = json.loads(r.content.decode('utf-8'))
            if data.get("object", False) == "error": 
                print("API respond an error to url : {0}".format(url))
                return False
            if data.get("has_more", None) and data.get("next_page", None):
                content = self.get_content(data["next_page"])
                data["data"] += content.get("data", [])
        return data
    
    def download_card(self, card, folder):
        """Telecharge l'image de la carte (dict format Scryfall) et créer son dict pour un folder donné
           Un contrôle sur le format du nom de fichier est effectué pour retirer les caractères interdits"""
        card_name = card.get("name", "UNKNOWN")
        card_id = card.get("id", 0)
        image_url = card.get("image_uris", {}).get("normal", False)
        set_code = folder
        phash = ""
        card_dict = {"name":card_name, "id":card_id, "url":image_url, "folder":set_code, "phash":phash}
        image_path = "{0}\\{1}\\{2}.jpg".format(self.directory, folder, re.sub(r'[\\/*?:"<>|]', "", card_name))
        is_dl = self.download_image(image_url, image_path)
        if not is_dl: return False
        else : return card_dict
    
    def download_image(self, url, image_path):
        """Télécharge une image à partir d'une url"""
        if not url or not image_path: 
            print("{0} invalide".format(url))
            return False
        if os.path.exists(image_path):
            return False
        sleep(0.1)
        try:
            #print("Début du téléchargement de {0} OK.".format(os.path.basename(image_path)))
            urllib.request.urlretrieve(url, image_path)
            return True
        except :
            print("problème lors du téléchargement de l'url : {0}".format(url))
            return False
        
    def download_missing_images(self, real_list, ref_list, folder):
        """Comparaison de listes d'id-Scryfall de cartes"""
        missing_cards = list(set(ref_list) - set(real_list))
        missing_cards_dict = []
        for card in missing_cards:
            url = "https://api.scryfall.com/cards/{0}".format(card)
            content = self.get_content(url)
            name = content.get("name","UNKNOWN")
            image_path = "{0}\\{1}\\{2}.{3}".format(self.directory, folder, content.get("name","UNKNOWN"), "jpg")
            url = content.get("image_uris", {}).get("normal", False)
            print("[{0}]Téléchargement de {1}".format(folder, name))
            self.download_image(url, image_path)
            card_dict = {"name":name, "id":content.get("id", 0), "url":url, "set":content["set"], "phash":""}
            missing_cards_dict.append(card_dict)
        return missing_cards_dict
    
    def download_set(self, edition):
        """Telecharge les images de l'édition (dict format Scryfall) donnée en paramètre
           Créer un dossier dédié avec son fichier json"""
        set_code = edition.get("code", "UNKNOWN")
        if set_code in self.forbidden_codes:
            set_code += "_"
        folder_path = self.directory + set_code
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        set_dict = {"set":edition.get("code", "UNKNOWN"), "total_cards":edition["card_count"],"cards":[]}
        content = self.get_content(edition.get("search_uri", False))
        if not content:
            print("[{0}]Erreur rencontrée lors du téléchargement du set.".format(set_code))
            return False
        for card in content.get("data", []):
            card_dict = self.download_card(card, set_code)
            if card_dict: set_dict["cards"].append(card_dict)
        
        with open(folder_path + r'\set_info.json', 'w') as f:
            json.dump(set_dict, f)    
        return True
    
    def get_hashes(self):
        folders = self.get_folders_list()
        hash_list = []
        for ind, folder in enumerate(folders):
            #print("[{0}/{1}]".format(ind, len(folders)))
            folder_path = os.path.join(self.directory, folder)
            json_file = os.path.join(folder_path, 'set_info.json')
            with open(json_file, 'r') as f:
                data = json.load(f)
            hash_list += [(card["phash"], card["name"], card["folder"]) for card in data["cards"] if card["phash"]!=""]
        return hash_list
    
    def hash_entity(self, path, level="card"):
        """Méthode récursive qui applique un perceptual hash sur une entité:
            directory, folder ou card"""
        
        if level == "card" and os.path.splitext(path)[1] == ".jpg": 
            try:
                img_phash = ih.phash(Image.open(path))
            except:
                print("pb de hashage sur la carte {0}".format(path))
                img_phash = ""
            return str(img_phash)
        
        elif level == "folder" and os.path.isdir(path):
            json_path = os.path.join(path, 'set_info.json')
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for card in data["cards"]:
                if card["phash"] != "": continue
                img_path = os.path.join(path, card["name"] + ".jpg")
                card["phash"] = self.hash_entity(os.path.join(path, img_path), level="card")
            
            with open(json_path, 'w') as f:
                json.dump(data, f)   
            return True
        
        elif level == "directory" and os.path.isdir(path):
            folders = self.get_folders_list()
            for ind, folder in enumerate(folders):
                print("[{0}/{1}] {2} en cours...".format(ind, len(folders), folder))
                self.hash_entity(os.path.join(path, folder), level="folder")
            return True
        
        else: return False
    
    def control_directory(self):
        """Méthode de contrôle les folders et leur contenu"""
        folders = self.get_folders_list()
        for ind, folder in enumerate(folders):
            path = os.path.join(self.directory, folder)
            if len(os.listdir(path)) == 0:
                print("[{0}] est vide, suppression.".format(folder))
                os.rmdir(path)
                continue
            set_code = folder.replace("_","")
            control_set = self.get_content("https://api.scryfall.com/sets/{0}".format(set_code))
            if control_set: 
                # Contrôle si le set est digital
                if control_set.get("digital", None): print("[{0}] est digital, à supprimer...")
                if not control_set.get("card_count", None):
                    print("[{0}] est vide, à supprimer...")
                    continue
                #control_set = self.get_content(control_set.get("search_uri", False))
            # Controle du nombre de cartes :
            with open(path + r'\set_info.json', 'r') as f:
                data = json.load(f)
                
            num = len(data["cards"])
            num_jpgs = len([file for file in os.listdir(path) if os.path.splitext(file)[1] == ".jpg"])
            expected_num =  control_set.get("card_count",0)
            if num_jpgs == int(expected_num) and num_jpgs == num:
                #print("[{0}] Tout est ok !".format(folder))
                continue
            else: print("[{0}/{1}][{2}] PB ! expected={3} ; jpg={4} ; json={5}".format(ind, len(folders), folder, expected_num, num_jpgs, num))

      
    def download_all_cards(self, digital=False):
        """Méthode permettant le téléchargement de toutes les images de cartes Magic"""
        sets = self.get_set_list()
        for ind, edition in enumerate(sets):
            set_code = edition.get("code", "UNKNOWN")
            if edition.get("digital", digital): 
                print("{0} est une édition digitale".format(set_code))
                continue
            print("[{0}/{1}] Téléchargement du set {2}.".format(ind, len(sets), set_code))
            self.download_set(edition)
        print("END")
        return True
    
    def update_directory(self):
        sets = self.get_set_list()
        folders = self.get_folders_list()
        for edition in sets:
            set_code = edition.get("code", "UNKNOWN")
            if not edition.get("digital", False) or not edition.get("card_count", None): 
                print("{0} est une édition digitale ou vide".format(edition.get("code", "UNKNOWN")))
                continue
            if set_code in self.forbidden_codes:
                set_code += "_"
            folder_path = os.path.join(self.directory, set_code)
            if not set_code in folders:
                # On télécharge le set manquant
                print("Nouveau set detecté ! [{0}] Téléchargement en cours".format(edition.get("code", "UNKNOWN")))
                self.download_set(edition)
            else:
                # On vérifie que le dossier de l'édition comporte le bon nombre de cartes
                expected_num = int(edition.get("card_count"))
                real_num = len(os.listdir(folder_path)) - 1
                dif = expected_num - real_num
                if dif != 0:
                    # Une différence est constaté, on cherche les cartes manquantes
                    print("[{0}] {1} carte(s) manquante(s), mise à jour en cours.".format(set_code, dif))
                    with open(folder_path + r'\set_info.json', 'r') as f:
                        data = json.load(f)
                    control_set = self.get_content(edition.get("search_uri", False))
                    real_list = [c["id"] for c in data["cards"]]
                    if control_set.get("cards", None): 
                        ref_list = [c["id"] for c in control_set.get("cards", None)]
                        missing_cards_dict = self.download_missing_images(real_list, ref_list, folder_path)
                        data["cards"] += missing_cards_dict
                        data["total_cards"] = len(data["cards"])
                        with open(folder_path + r'\set_info.json', 'r') as f:
                            json.dump(data)
                    else: print("[{0}] Cartes introuvables pour ce set.".format(set_code))
        return True
    
    def reverse_update(self):
        folders = self.get_folders_list()
        for folder in folders:
            folder_path = os.path.join(self.directory, folder)
            images = [os.path.splitext(file)[0] for file in os.listdir(folder_path) if os.path.splitext(file)[1] == ".jpg"]

            try:            
                with open(folder_path + r'\set_info.json', 'r') as f:
                    data = json.load(f)
                names = [(re.sub(r'[\\/*?:"<>|]', "", card.get("name", "UNKNOWN")), card.get("url", None)) for card in data["cards"]]
            except:
                print("[{0}] Problème de données sur ce set.".format(folder))
                continue
            dif = len(set(map(itemgetter(0), names))) - len(set(images))
            print("[{0}] Différence de {1} carte(s) constaté sur ce set.".format(folder, dif))
            for name, url in names:
                if name not in images:
                    image_path = os.path.join(folder_path, name + ".jpg")
                    self.download_image(url, image_path)
    
    def reload_all(self):
        sets = self.get_set_list()
        for edition in sets:
            if edition.get("code", "UNKNOWN") != "10e": continue
            # Contrôle si édition vide ou digitale
            if edition.get("card_count", 0) == 0 or edition.get("digital", False): continue
            folder = edition.get("code", "UNKNOWN")
            print(folder)
            # Contrôle si édition possède un nom de dossier interdit
            if folder in self.forbidden_codes:
                folder += "_"
            folder_path = os.path.join(self.directory, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            control_set = self.get_content(edition.get("search_uri", False))
            # Contrôle si le chargement des cartes du set est ok
            if not control_set.get("data", False):
                print("[{0}] API Erreur dans le chargement du set (search_uri).".format(folder))
                continue
            
            set_dict = {"folder":folder, "name":control_set.get("name", "UNKNOWN"), "cards":[]}
            basic_lands = {}
            for card in control_set["data"]:
                # Contrôle si la carte est un land de base
                if "Basic Land" in card.get("type_line", "UNKNOWN"):
                    card_name = re.sub(self.forbidden_chars, "", card.get("name", "UNKNOWN"))
                    if not basic_lands.get(card_name, False):
                        basic_lands[card_name] = [card]
                    else: basic_lands[card_name].append(card)
                    continue
                card_name = re.sub(self.forbidden_chars, "", card.get("name", "UNKNOWN"))
                filename = os.path.join(folder_path, card_name + ".jpg")
                card_id = card.get("id", "UNKNOWN")
                image_url = card.get("image_uris", {}).get("normal", False)
                phash = ""
                # Contrôle si la carte est déjà téléchargée
                card_dict = {"name":card_name, "id":card_id, "url":image_url, "folder":folder, "phash":phash}
                self.download_image(image_url, filename)
                set_dict["cards"].append(card_dict)
                
            # ON effectue un traitement spécial pour les basics lands
            for basic in basic_lands:
                for ind, card in enumerate(basic_lands[basic]):
                    card_id = card.get("id", "UNKNOWN")
                    image_url = card.get("image_uris", {}).get("normal", False)
                    card_name = re.sub(self.forbidden_chars, "", card.get("name", "UNKNOWN"))
                    filename = os.path.join(folder_path, card_name + ".jpg")
                    # Si le basic land est déjà là on le supprime pour retelecharger proprement
                    if os.path.exists(filename):
                        os.remove(filename)
                    card_name = "{0} {1}".format(card_name, ind + 1)
                    filename = os.path.join(folder_path, card_name + ".jpg")
                    card_dict = {"name":card_name, "id":card_id, "url":image_url, "folder":folder, "phash":phash}
                    self.download_image(image_url, filename)
                    set_dict["cards"].append(card_dict)

            with open(os.path.join(folder_path,'set_info.json'), 'w') as f:
                json.dump(set_dict, f) 
    
    def dl_set_icons(self):
        mv_url =  "http://www.magic-ville.com/fr"
        set_numbers = range(-1,290)
        for set_num in set_numbers:
            #if set_num != 0: continue
            url = "{0}/set_cards?setcode={1}".format(mv_url, set_num)
            sleep(0.1)
            r = requests.get(url)
            if not r.status_code == requests.codes.ok: 
                print("Problème avec le set n°{0}".format(set_num))
                continue
            c = r.content
            soup = BeautifulSoup(c, 'html.parser')
            if not soup.title: continue
            t = soup.title.string
            t = t.split(" - ")[0]
            if "/" in t:
                t = t.split("/")[-1].strip()
            set_name = re.sub(self.forbidden_chars, "", t)
            if set_name in self.forbidden_codes:
                print("Impossible de créer un dossier avec ce nom: {0}".format(set_name))
                continue
            folder_path = os.path.join(self.directory, set_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            images = soup.find_all("img", src=re.compile(r'symbols'))
            if len(images) == 0: print("{0} have no set symbols !")
            print(set_name)
            for image in images:
                image_name = re.sub(self.forbidden_chars, "", os.path.basename(image['src']))
                image_path = os.path.join(folder_path, image_name)
                image_url = "{0}/{1}".format(mv_url, image['src'])
                self.download_image(image_url, image_path)

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
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

def get_framed_card(photo):
    ratio = photo.shape[0] / 300.0
    orig = photo.copy()
    photo = resize(photo, height = 300)
    
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
    

    #warp = exposure.rescale_intensity(warp, out_range = (0, 255))

    return warp

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product
    
def score_images(im1, im2, method):
    """im1 = photo ; im2 = ref_im"""
    score = 0
    if method == "ssim":
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        score = ssim(cv2.resize(im1, (im2.shape[1], im2.shape[0]), interpolation = cv2.INTER_AREA), im2, multichannel=False)
    if method == "hist_inter":
        photo_hist = cv2.calcHist([cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0,256])
        gray_card_im = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        image_hist = cv2.calcHist([gray_card_im], [0], None, [256], [0,256])
        score = cv2.compareHist(photo_hist, image_hist, method = cv2.HISTCMP_INTERSECT)
    if method == "cor":
        #im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        #im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        im1 = cv2.resize(im1, (im2.shape[1], im2.shape[0]))
        #score = cv2.matchTemplate(im2, im1, cv2.TM_CCOEFF_NORMED)
        score = correlation_coefficient(im1, im2)
        # print(score)
    return score

def get_best_match(photo, phashes):
    #split = cv2.split(warp)
    photo_array = Image.fromarray(photo)
    phash_photo = str(ih.phash(photo_array))
    distances = sorted([(distance.hamming(phash_photo, i[0]), i[1], i[2]) for i in phashes], key=itemgetter(0))
    
    comparaisons = []
    for im_phash, name, folder in distances[:20]:
        path = "{0}\{1}\{2}.jpg".format(r"d:\Profiles\ncapon\Desktop\Perso\MTG_IMG", folder, name)
        comp_image = cv2.imread(path)
        result = score_images(photo, comp_image, method="cor")
        comparaisons.append((result, name, folder))
    best_match = max(comparaisons, key = itemgetter(0))

    return sorted(comparaisons, key=itemgetter(0))
   
def test():             
    directory = r"d:\Profiles\ncapon\Desktop\Perso\MTG_IMG"
    iM = ImageManager(directory)
    print("Ready")
    phashes = iM.get_hashes()
    path_photo = "D:\\Profiles\\ncapon\\Desktop\\Perso\\MTG_photo\\"
    print("_____________________")
    s = time.time()
    filename = path_photo + "photo9.jpg"
    photo = cv2.imread(filename)
    card_photo = get_framed_card(photo)
    print(get_best_match(card_photo, phashes))
    e = time.time()
    print(e - s)
    print("_____________________")
    s = time.time()
    filename = path_photo + "photo10.jpg"
    photo = cv2.imread(filename)
    card_photo = get_framed_card(photo)
    print(get_best_match(card_photo, phashes))
    e = time.time()
    print(e - s)
    print("_____________________")
    s = time.time()
    filename = path_photo + "photo11.jpg"
    photo = cv2.imread(filename)
    card_photo = get_framed_card(photo)
    print(get_best_match(card_photo, phashes))
    e = time.time()
    print(e - s)
    print("_____________________")
    s = time.time()
    filename = path_photo + "photo12.jpg"
    photo = cv2.imread(filename)
    card_photo = get_framed_card(photo)
    print(get_best_match(card_photo, phashes))
    e = time.time()
    print(e - s)
    # iM.hash_entity(directory, level="directory")
    print("END")

directory = r"d:\Profiles\ncapon\Desktop\Perso\MTG_ICON"
iM = ImageManager(directory)
iM.dl_set_icons()