import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# dataset_path = "D:/Python projects/Classification/Dataset"
#
# categories = ['Jasmine', 'Lemon', 'Mint', 'Neem', 'Peepal']
# data = []
#
# for category in categories:
#     path = os.path.join(dataset_path, category)
#     label = categories.index(category)
#
#     for img in os.listdir(path):
#         imgpath = os.path.join(path, img)
#         herb_img = cv2.imread(imgpath, 0)
#         herb_img = cv2.resize(herb_img, (224, 224))
#         # cv2.imshow('Image', herb_img)
#         image = np.array(herb_img).flatten()
#
#         data.append([image, label])
# # print(len(data))
#
# # making a pickle of loaded images
# pick_in = open('dataset_info.pickle', 'wb')
# pickle.dump(data, pick_in)
# pick_in.close()

# ----------------------------------------------------


pick_in = open('D:/Python projects/Classification/dataset_info.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

# splitting dataset
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.25)

# training model
model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

categories = ['Jasmine', 'Lemon', 'Mint', 'Neem', 'Peepal']

print('Accuracy = ', accuracy)
print('Prediction is = ', categories[prediction[0]])

herb = xtest[0].reshape(224, 224)

plt.imshow(cv2.cvtColor(herb, cv2.COLOR_GRAY2BGR))
# plt.imshow(herb)
plt.show()








