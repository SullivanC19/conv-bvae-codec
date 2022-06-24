import torch
from torchvision import datasets, transforms
import cv2
import os
import numpy as np
import math
import random

DIR_DATA = './res/data'

DIR_FACES = './res/data/colin_faces/'
DIR_SMILING_FACES = './res/data/smiling_colin_faces/'

F_MODEL = './models/convbvae_full_conv_w_fc_lat_32_gaussvar_normalized_tr_1e5.pkl'
F_FACE_CLASSIFIER = './res/haarcascade_frontalface_alt_tree.xml'
F_VIDEO = './res/snapshots.avi'

R_SEED = 42

CROP_SCALE = .7
CROP_VERT_SHIFT = 30

IMG_SIZE = (512, 512)

TRAIN_SET_SIZE = 6000
TEST_SET_SIZE = 2000

def load_image_data(img_dir, batch_size=16, input_img_size=64):
    transform = transforms.Compose([
        transforms.Resize(input_img_size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    dataset = datasets.ImageFolder(
        root=img_dir,
        transform=transform
    )

    test_dataset, train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [TEST_SET_SIZE, TRAIN_SET_SIZE, len(dataset) - (TEST_SET_SIZE + TRAIN_SET_SIZE)], generator=torch.Generator().manual_seed(R_SEED))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return test_loader, train_loader, valid_loader

def crop_and_shift(img):
    h, w, _ = img.shape
    size = int(min(h, w) * CROP_SCALE)
    r1, r2 = h // 2 - size // 2 + CROP_VERT_SHIFT, h // 2 + size // 2 + CROP_VERT_SHIFT
    c1, c2 = w // 2 - size // 2, h // 2 + size // 2
    return img[r1:r2, c1:c2, :]

def compare_encoded_decoded(disp_img_size=(200, 100)):
    data = load_image_data(DIR_DATA, batch_size=1)
    model = torch.load(F_MODEL)

    for images, _ in data:
        x = images
        img_in = from_output_to_image(x)

        mean, _ = model.encoder(x)
        z = mean # no randomness in sample here for the sake of consistency
        x_hat = model.decoder(z)
        img_out = from_output_to_image(x_hat)

        img = np.concatenate((img_in, img_out), axis=1)
        cv2.imshow('image', cv2.resize(img, disp_img_size))

        # continue on keypress and quit on 'q'
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def sample_animation(radius=50, latent_var_idx=0, rot_per_sec=20):
    model = torch.load(F_MODEL)

    i = 0
    while True:
        theta = torch.tensor(2 * math.pi * (i / rot_per_sec))
        z = torch.zeros((1, model.latent_variables))
        z[0, latent_var_idx] = torch.sin(theta) * radius
        x = model.decoder(z)
        cv2.imshow('image', from_output_to_image(x))

        # short delay or quit on 'q'
        key = cv2.waitKey(100)
        if key == ord('q'):
            break

        i = (i + 1) % rot_per_sec

# permute from numpy ourput to RGB and shift from [-1, -1] to [0, 255]
def from_output_to_image(x):
    img = np.array((x.detach().permute(0, 2, 3, 1).numpy()*0.5+0.5)*255, dtype=np.uint8)
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

def generate_images_from_vid():
    # construct face detector: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
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

        # detect faces using HAAR cascade
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
        cropped = crop_and_shift(face_img)
        # 4. downscale
        img = cv2.resize(cropped, IMG_SIZE)

        cv2.imwrite(DIR_FACES + f'{i}.png', img)

        i += 1
    
    vid_capture.release()

def prune_and_label_images(num_images=8200):
    quit = False
    while True:
        i = random.randint(0, num_images - 1)

        f_img = DIR_FACES + f'{i}.png'
        f_smiling_img = DIR_SMILING_FACES + f'{i}.png'
        if not os.path.exists(f_img):
            continue

        img = cv2.resize(cv2.imread(f_img), (400, 400))
        img = cv2.putText(img, str(i), (0, 50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255))
        cv2.imshow('image', img)

        while True: # loop until a valid key ('q', 'r', 's', or 'n') is pressed
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'): # quit
                quit = True
                break
            if key == ord('r'): # remove (prune) image
                os.remove(f_img)
                break
            if key == ord('s'): # label smiling image
                os.rename(f_img, f_smiling_img)
                break
            if key == ord('n'): # next
                break
        
        if quit:
            break

        i += 1

