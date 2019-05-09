import cv2
#input image change
img = cv2.imread('algus.png', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
width = 64
height = 64
dim = (width, height)
 
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imwrite( "algus1.png", resized );