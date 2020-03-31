import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from test import evaluate
from skimage.filters import gaussian

# plt.switch_backend("qt5Agg")
plt.switch_backend("tkAgg")
# TODO: Fix the dictionry
SEGMENTS = {
    0: "background",
    1: "skin",
    2: "r_brow",
    3: "l_brow",
    4: "r_eye",
    5: "l_eye",
    10: "nose",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    18: "hat",
}


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 250, 250]):
    b, g, r = color  # [10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]

    return changed


def change_color(image, parsed_mask, **kwargs):
    """

   :param image:
   :param parsed_mask:
   :param query:
   :return:

   Query (kwargs) example:

   {
       'background': (R, G, B)
       'neck': (R, G, B)
       'skin': (R, G, B)
       'hat': (R, G, B)
       'nose': (R, G, B)
       'l_eye': (R, G, B)
       'r_eye': (R, G, B)
       'u_lip': (R, G, B)
       'l_lip': (R, G, B)
       'l_brow': (R, G, B)
       'r_brow': (R, G, B)
    }
   """
    # Permuting color spaces form RGB to BGR
    query = {SEGMENTS[key]: color for key, color in kwargs.items()}
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for key, color in query.items():
        b, g, r = color
        # Allocate mask
        mask = np.zeros_like(image)
        mask[:, :, 0] = b
        mask[:, :, 1] = g
        mask[:, :, 2] = r

        if key == 12 or key == 13:
            image_hsv[:, :, 0:2] = target_hsv[:, :, 0:2]

        else:
            image_hsv[:, :, 0:1] = target_hsv[:, :, 0:1]

        new_image = sharpen(cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR))

        if key == 17:
            new_image = sharpen(new_image)


        new_image[parsed_mask != key] = image[parsed_mask != key]

        return new_image

if __name__ == "__main__":
    # 1  face
    # 11 teeth
    # 12 upper lip
    # 13 lower lip
    # 17 hair

    """
     0: 'background'
     14: 'neck'
     1: 'skin'
     18: 'hat'
     10: 'nose'
     5: 'l_eye'
     4: 'r_eye'
     12: 'u_lip'
     13: 'l_lip'
     3: 'l_brow'
     2: 'r_brow'
     
     
      8: 'l_ear'
     9: 'r_ear'
     10: 'mouth'
     17: 'hair'
     15: 'ear_r'
     16: 'neck_l'
     18: 'cloth'
     3: 'eye_g'
 """
    parse = argparse.ArgumentParser()
    parse.add_argument("--img-path", default="imgs/before.jpg")
    args = parse.parse_args()

    table = {
        "hair": 17,
        "upper_lip": 12,
        "lower_lip": 13,
    }

    image_path = "/home/aziz/Projects/face/imgs/6.jpg"
    cp = "cp/79999_iter.pth"

    image = cv2.imread(image_path)
    ori = image.copy()
    parsing = evaluate(image_path, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    parts = [
        table["hair"],
        table["lower_lip"],
        table["upper_lip"],
    ]

    alpha_slider_max = 255
    title_window = "Linear Blend"

    change_color(image, parsing, u_lip=(-1, 0, 255))
    for i in range(2):
        image = cv2.imread(image_path)

        lips = np.random.randint(1, 255, (3))
        hair_ = np.random.randint(1, 255, (3))
        colors = np.array([hair_, lips, lips])

        for part, color in zip(parts, colors):
            image = hair(image, parsing, part, color)

        # kernel = np.ones((5, 5), np.float32) / 25
        # dst = cv.filter2D(image, -1, kernel)
        dst = cv2.bilateralFilter(image, 30, 75, 75)

        img = np.hstack((ori, dst))
        plt.imshow(cv2.cvtColor(cv2.resize(img, (2048, 1024)), cv2.COLOR_BGR2RGB))
        plt.show()
        # cv2.imwrite("makeup.jpg", cv2.resize(img, (1536, 512)))

        # cv2.imshow('color', cv2.resize(image, (512, 512)))
        # cv2.imwrite('image_1.jpg', cv2.resize(ori, (512, 512)))
        # cv2.imwrite('makeup.jpg', cv2.resize(img, (1536, 512)))

    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        print("killed")
    cv2.destroyAllWindows()
