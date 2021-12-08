import torch
import cv2
import os
import numpy as np
import math
import random

from trainer import load_image_data, DIR_DATA

F_MODEL = './models/convbvae_full_conv_w_fc_lat_32_gaussvar_normalized_tr_1e5.pkl'
F_FACE_CLASSIFIER = './res/haarcascade_frontalface_alt_tree.xml'
F_VIDEO = './res/snapshots.avi'
DIR_FACES = './res/data/colin_faces/'
DIR_SMILING_FACES = './res/data/smiling_colin_faces/'
CROP_SCALE = .7
CROP_VERT_SHIFT = 30
SIZE = 512

def crop(img, scale, vert_shift):
    h, w, _ = img.shape
    size = int(min(h, w) * scale)
    r1, r2 = h // 2 - size // 2 + CROP_VERT_SHIFT, h // 2 + size // 2 + CROP_VERT_SHIFT
    c1, c2 = w // 2 - size // 2, h // 2 + size // 2
    return img[r1:r2, c1:c2, :]

def compare_encoded_decoded():
    data = load_image_data(DIR_DATA, batch_size=1)
    model = torch.load(F_MODEL)
    i = 0
    for images, _ in data:
        x = images
        img_in = from_output_to_image(x)

        mean, log_var = model.encoder(x)
        std = torch.exp(log_var / 2)

        # sample z using reparameterization trick
        eps = torch.normal(torch.zeros_like(mean))
        z = mean

        # decode using sampled z
        x_hat = model.decoder(z)
        img_out = from_output_to_image(x_hat)

        img = np.concatenate((img_in, img_out), axis=1)
        cv2.imshow('image', cv2.resize(img, (200, 100)))
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def sample_animation():
    model = torch.load(F_MODEL)
    i = 0
    while True:
        rad = torch.tensor(2 * math.pi * (i / 20))
        z = torch.zeros((1, 32))
        imgs = []
        for j in range(1):
            z[j][j] = torch.sin(rad) * 50
        x = model.decoder(z)
        cv2.imshow('image', from_output_to_image(x))#cv2.resize(from_output_to_image(x), (50 * 32,50)))
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break

        i += 1


def from_output_to_image(x):
    arr = np.concatenate(np.array((x.detach().permute(0, 2, 3, 1).numpy()*0.5+0.5)*255, dtype=np.uint8), axis=1)
    return cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)

def generate_images():
    # construct face detector
    # credit: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
    face_detector = cv2.CascadeClassifier(F_FACE_CLASSIFIER)

    # start video capture
    vid_capture = cv2.VideoCapture(F_VIDEO)
    assert vid_capture.isOpened()

    # get frame count for sanity check
    fps = vid_capture.get(cv2.CAP_PROP_FPS)
    frame_count = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f'FPS: {fps}')
    print(f'# of Frames: {frame_count}')

    i = 0
    while vid_capture.isOpened():
        success, frame = vid_capture.read()
        if not success:
            break

        # detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.01, 5)

        if len(faces) == 0:
            i += 1
            continue

        # 1. get best face capture
        (x, y, w, h) = faces[0]
        # 2. capture face
        face_img = frame[y:y+h, x:x+w, :]
        # 3. crop
        cropped = crop(face_img, CROP_SCALE, CROP_VERT_SHIFT)
        # 4. downscale
        img = cv2.resize(cropped, (SIZE, SIZE))
        cv2.imwrite(DIR_FACES + f'{i}.png', img)

        i += 1
    
    vid_capture.release()

def check_images(num_images=8200):
    quit = False
    # i = 0
    while True:
        i = random.randint(0, num_images - 1)

        f_img = DIR_FACES + f'{i}.png'
        f_smiling_img = DIR_SMILING_FACES + f'{i}.png'
        if not os.path.exists(f_img):
            continue

        img = cv2.resize(cv2.imread(f_img), (400, 400))
        img = cv2.putText(img, str(i), (0, 50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255))
        cv2.imshow('image', img)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                quit = True
                break
            if key == ord('r'):
                os.remove(f_img)
                break
            if key == ord('s'):
                os.rename(f_img, f_smiling_img)
                break
            # if key == ord('b'):
            #     i -= 2
            #     break
            if key == ord('n'):
                break
        
        if quit:
            break

        i += 1

if __name__ == '__main__':
    check_images()
    # generate_images()
    # sample_animation()
    # compare_encoded_decoded()

