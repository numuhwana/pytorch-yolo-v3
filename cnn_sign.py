from keras.models import load_model
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from PIL import Image
from keras.preprocessing import image

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def loadsignmodel():
    model = load_model('./weight_sign/traffic_sign_v1.h5')
    model.compile(loss='binary_crossentropy',   # 최적화 함수 지정
    optimizer='adam',
    metrics=['accuracy'])
    Cu=Image.open('./signpic/caution.png','r')
    Cu=np.array(Cu)
    Cu=Cu[...,::-1]
    In=Image.open('./signpic/indication.png','r')
    In=np.array(In)
    In=In[...,::-1]
    Rg=Image.open('./signpic/regulation.png','r')
    Rg=np.array(Rg)
    Rg=Rg[...,::-1]
    return model,Cu,In,Rg

def predictsign(model,test_img,Cu,In,Rg):
    categories = [Rg,Cu,In]
    test_image = np.expand_dims(test_img, axis = 0)

    pred = model.predict_classes(test_image)
    return categories[pred[0]]


