import cv2
import numpy as np
import random
import os

class Image(object):
    def __init__(self, id):
        self.id = id
        self.rust = None
        self.bg = None
        self.masks = list()
        self.mask = None
        self.image = None
        self.image_shape = None

    def load_image(self, rust_path, bg_path):
        self.rust = cv2.imread(rust_path)
        self.rust = cv2.resize(self.rust, (1300,866))
        self.image_shape = self.rust.shape

        self.bg = cv2.imread(bg_path)
        self.bg = cv2.resize(self.bg, (self.image_shape[1], self.image_shape[0]))
        
        self.mask = np.zeros(self.image_shape[:2], dtype=np.uint8)
    
    def get_random_loc(self, radius):
        locx = random.randrange(radius, self.image_shape[0] - radius)
        locy = random.randrange(radius, self.image_shape[1] - radius)
        return locx, locy

    def get_random_point(self, locx, locy, sigma_x, sigma_y, threshold):
        ptx = np.random.normal(locx, sigma_x)
        pty = np.random.normal(locy, sigma_y)
        if ptx <= 0: ptx = 0
        if ptx >= self.image_shape[1]: ptx = self.image_shape[1]
        if pty <= 0: pty = 0
        if pty >= self.image_shape[0]: pty = self.image_shape[0]
        return ptx, pty

    def get_point_list(self, sigma_x, sigma_y, radius, num_points, kernel, locx, locy):
        ptsx = list()
        ptsy = list()
        for j in range(num_points):
            ptx, pty = self.get_random_point(locx, locy, sigma_x, sigma_y, radius)
            ptsx.append(ptx)
            ptsy.append(pty)
        return ptsx, ptsy

    def draw_dots(self, ptsx, ptsy, radius, mask):
        for ptx, pty in zip(ptsx, ptsy):
            cv2.circle(mask, (int(ptx), int(pty)), radius, (255,255,255), -1)
        return mask

    def morph_dots(self, kernel, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        return mask
        

    def create_image(self, rust_path, bg_path, sigmas_x = (10, 20, 40, 60, 100), sigmas_y = (10, 20, 40, 60, 100), 
                     radius = 2, num_locs = 5, num_points = 50, 
                     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 30))):
        #Loading image
        self.load_image(rust_path, bg_path)
        #Randomly altering mask to show rust spots
        for i in range(num_locs):
            sigma_x = random.choice(sigmas_x)
            sigma_y = random.choice(sigmas_y)
            locx, locy = self.get_random_loc(radius)
            ptsx1, ptsy1 = self.get_point_list(sigma_x, sigma_y, radius, num_points, kernel, locx, locy)

            #Creating main rust area
            mask = np.zeros(self.image_shape[:2], dtype=np.uint8)
            mask = self.draw_dots(ptsx1, ptsy1, radius, mask)
            mask = self.morph_dots(kernel, mask)

            #Adding some noise
            ptsx2, ptsy2 = self.get_point_list(sigma_x, sigma_y, radius, num_points*2, kernel, locx, locy)
            mask = self.draw_dots(ptsx2, ptsy2, radius - 1, mask)

            ptsx3, ptsy3 = self.get_point_list(sigma_x, sigma_y, radius, num_points*10, kernel, locx, locy)
            mask = self.draw_dots(ptsx3, ptsy3, radius - 2, mask)
            
            self.masks.append(mask)

        #Adding images
        for mask in self.masks:
            self.mask = cv2.add(self.mask, mask)
        masked_rust = cv2.bitwise_and(self.rust, self.rust, mask=self.mask)
        inv_mask = cv2.bitwise_not(self.mask)
        masked_bg = cv2.bitwise_and(self.bg, self.bg, mask=inv_mask)

        self.image = cv2.add(masked_rust, masked_bg)

def generate_images(num_images, subset):
    #Creating directories
    image_dir = 'datasets/generated_dataset/{}/images'.format(subset)
    masks_dir = 'datasets/generated_dataset/{}/masks'.format(subset)

    rust_dir = 'C:/Users/Fredrik/Desktop/Mask_RCNN/datasets/generated_dataset/rust_images' 
    bg_dir = 'C:/Users/Fredrik/Desktop/Mask_RCNN/datasets/generated_dataset/bg_images' 
    
    if not os.path.exists(image_dir):
            os.makedirs(image_dir)
    if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)

    for i in range(num_images):
        #Creating image
        image = Image(i)

        rust_path = '{}/{}'.format(rust_dir, random.choice(os.listdir(rust_dir)))
        bg_path = '{}/{}'.format(bg_dir, random.choice(os.listdir(bg_dir)))
        sigmas = (10, 20, 30, 40, 50, 80)
        image.create_image(rust_path, bg_path, 
                            sigmas_x = sigmas, sigmas_y = sigmas,
                            radius = 3, num_locs = 5, num_points = 100, 
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 12)))
        
        #Writing images to images directory
        cv2.imwrite('{}/{}.jpg'.format(image_dir, str(image.id)), image.image)

        mask_dir = '{}/{}'.format(masks_dir, str(image.id))
        #Creating mask directories
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        for i, mask in enumerate(image.masks):
            cv2.imwrite('{}/{}.jpg'.format(mask_dir, str(i)), mask)

def generate_test_images(num_images, subset):
    #Creating directories
    image_dir = 'datasets/generated_dataset/test/{}/images'.format(subset)
    masks_dir = 'datasets/generated_dataset/test/{}/masks'.format(subset)
    
    if not os.path.exists(image_dir):
            os.makedirs(image_dir)
    if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)

    for i in range(num_images):
        #Creating images
        image = Image(i)
        image.create_image('C:/Users/Fredrik/Desktop/Mask_RCNN/datasets/generated_dataset/rust_images/rust1.jpg', 
                           'C:/Users/Fredrik/Desktop/Mask_RCNN/datasets/generated_dataset/bg_images/bg2.jpg',
                           sigmas = (10, 20, 30, 40, 50, 60), radius = 3, num_locs = 5, num_points = 1500, 
                           kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 30)))
        #Writing images to images directory
        cv2.imwrite('datasets/generated_dataset/test/{}/images/{}.jpg'.format(subset, str(image.id)), image.image)

        #Creating mask directories
        mask_dir = 'datasets/generated_dataset/test/{}/masks/{}'.format(subset, str(image.id))
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        for i, mask in enumerate(image.masks):
            cv2.imwrite('{}/{}.jpg'.format(mask_dir, str(i)), mask)

def test_params():
    #Creating images
        image = Image(1)
        image.create_image('C:/Users/Fredrik/Desktop/Mask_RCNN/datasets/generated_dataset/rust_images/rust1.jpg', 
                           'C:/Users/Fredrik/Desktop/Mask_RCNN/datasets/generated_dataset/bg_images/bg2.jpg',
                           sigmas_x = (10, 20, 30, 40, 50, 60), sigmas_y = (10, 20, 30, 40, 50, 60), radius = 3, num_locs = 5, num_points = 1500, 
                           kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 30)))
        cv2.imshow('image', image.image)
        cv2.imshow('mask', image.mask)
        cv2.waitKey(-1)


#Generating images for subsets: train/val
generate_images(100, 'val')
#test_params()
