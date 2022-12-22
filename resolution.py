import cv2
import time

image=cv2.imread(r'D:\PROJECTs\super_resolution(python)\text.jpg')
width=image.shape[1]
height=image.shape[0]
bicubic=cv2.resize(image,(width*4,height*4))
cv2.imshow('test-image',image)
cv2.imshow('bicubic',bicubic)

super_res = cv2.dnn_superres.DnnSuperResImpl_create()

start = time.time()
super_res.readModel('super_resolution(python)\LapSRN_x4.pb')
super_res.setModel('lapsrn',4)
lapsrn_image = super_res.upsample(image)
end = time.time()
print('Time taken in seconds by lapsrn', end-start)
cv2.imshow('LAPSRN',lapsrn_image)




start = time.time()
super_res.readModel('super_resolution(python)\EDSR_x4.pb')
super_res.setModel('edsr',4)
edsr_image = super_res.upsample(image)
end = time.time()
print('Time taken in seconds by edsr', end-start)
cv2.imshow('EDSR',edsr_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

