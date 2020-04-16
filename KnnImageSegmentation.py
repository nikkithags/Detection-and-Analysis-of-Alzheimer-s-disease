image = cv2.imread('brain1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixel_values = image.reshape((-1, 3)) #2D and 3 colors
pixel_values = np.float32(pixel_values) #making the matrix float
#print(pixel_values.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2) #define the criteria
_, labels, centers = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) #asking for 3 classes - black, white and grey
centers = np.uint8(centers)
labels = labels.flatten()
final_image = centers[labels.flatten()]

final_image = final_image.reshape(image.shape)

plt.subplot(121)
plt.imshow(image)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(final_image)
plt.title('KNN (3) Image'), plt.xticks([]), plt.yticks([])
plt.show()
