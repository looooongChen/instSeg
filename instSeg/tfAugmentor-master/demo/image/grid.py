import cv2

img_path = './plant.png'
save_path = './plant_grid.png'
distance = 100

img = cv2.imread(img_path)

H, W = img.shape[0], img.shape[1]

y = distance
while y < H:
    cv2.line(img, (0, y), (W, y), color=(0,0,255), thickness=2)
    y += distance

x = distance
while x < W:
    cv2.line(img, (x, 0), (x, H), color=(0,0,255), thickness=2)
    x += distance

cv2.imwrite(save_path, img)
