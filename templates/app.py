import cv2


def is_mirror_image(img_path):
    img = cv2.imread(img_path)
    flipped = cv2.flip(img, 1)
    diff = cv2.subtract(img, flipped)
    b, g, r = cv2.split(diff)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        return True
    return False

f=f'50-swift-car-number-plate-frame-.webp'
if is_mirror_image(f):
    print("True")
else:
    print("False")