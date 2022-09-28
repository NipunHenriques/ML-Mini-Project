import os
import gradio as gr
import pickle
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, recall_score, precision_score


#dir = 'D:\\Documents\\College\\ML\\ML_Mini_Project\\kagglecatsanddogs_3367a\\PetImages'

#categories = ['Cat','Dog']
#data = []

#for category in  categories:
#    path = os.path.join(dir,category)
#    label = categories.index(category)
    
#    for img in os.listdir(path):
#        imgpath= os.path.join(path,img)
#        pet_img=cv2.imread(imgpath,0)
#        try:
#            pet_img=cv2.resize(pet_img,(50,50))
#            image = np.array(pet_img).flatten()
#            
#            data.append([image,label])
#        except Exception as e:
#            pass
        

#pick_in = open('data1.pickle','wb')
#pickle.dump(data,pick_in)
#pick_in.close()
def start(ML_mini_project):
        pick_in = open('data1.pickle','rb')
        data = pickle.load(pick_in)
        pick_in.close()

        random.shuffle(data)
        features = []
        labels = []

        for feature,label in data:
            features.append(feature)
            labels.append(label)

            
        xtrain, xtest, ytrain, ytest = train_test_split(features,labels, test_size = 0.05)

#model = SVC(C=1, kernel = 'poly', gamma = 'auto')
#model.fit(xtrain,ytrain)

#pick = open('model.sav','wb')
#pickle.dump(model,pick)
#pick.close()

        pick = open('model.sav','rb')
        model =pickle.load(pick)
        pick.close()

        categories = ['Cat','Dog']
        prediction = model.predict(xtest)
        confusion = confusion_matrix(ytest, prediction)
        precision = precision_score(ytest, prediction)
        recall = recall_score(ytest, prediction) 
        accuracy  = model.score(xtest, ytest)

        mypet=xtest[0].reshape(50,50)
        plt.imshow(mypet,cmap = 'gray')
        plt.show()
        print ("Accuracy:\n " ,accuracy)
        print ("Confusion Matrix\n", confusion)
        print("Precision: \n", precision)
        print("Recall: \n", recall)
        
        return "Prediction: "+categories[prediction[0]]
        
        

        

face = gr.Interface(fn= start, inputs = "text", outputs = "text" )
face.launch()
