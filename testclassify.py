#keras에서는 폴더 이름에 따라서 classification 이름이 정해진다
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def load_net():
    # load json and create model
    json_file = open('./weight_traffic/model_trafficv4.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./weight_traffic/model_trafficv4.h5")
    print("Loaded model from disk")


    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    Gf=Image.open('./lightpic/green.png','r')
    Gf=np.array(Gf)
    Gf=Gf[...,::-1]
    Rf=Image.open('./lightpic/red.png','r')
    Rf=np.array(Rf)
    Rf=Rf[...,::-1]

    return loaded_model,Gf,Rf


def classify_traffic(test_image,model,Gf,Rf):
    #single test
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict_classes(test_image)
    prediction=''
    predpic=Gf
    if result[0][0] == 1:## Red light일때
        #prediction = 'Redlight'
        predpic=Rf
    #else:
        #prediction = 'Greenlight'

    #print(prediction)
    return predpic


