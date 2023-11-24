from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
# from pushbullet import PushBullet
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image


# Loading Models

diabetes_model = pickle.load(open('models/diabetes.sav', 'rb'))
heart_model = pickle.load(open('models/heart_disease.pickle.dat', "rb"))
braintumor_model = load_model('models/braintumor.h5')
alzheimer_model = load_model('models/alzheimer_model.h5')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Configuring Flask

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True
model=load_model('models/ECG.h5')#loading the model

app.secret_key = "secret key"

def preprocess_imgs(set_name, img_size):
#     """
#     Resize and apply VGG-15 preprocessing
#     """
    set_new = []
    for img in set_name:
          img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
          set_new.append(preprocess_input(img))
          return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
#     """
#     Finds the extreme points on the image and crops the rectangular out of them
#     """
    set_new = []
    for img in set_name:
          gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
          gray = cv2.GaussianBlur(gray, (5, 5), 0)

#         # threshold the image, then perform a series of erosions +
#         # dilations to remove any small regions of noise
          thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
          thresh = cv2.erode(thresh, None, iterations=2)
          thresh = cv2.dilate(thresh, None, iterations=2)

#         # find contours in thresholded image, then grab the largest one
          cnts = cv2.findContours(
              thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          cnts = imutils.grab_contours(cnts)
          c = max(cnts, key=cv2.contourArea)

#         # find the extreme points
          extLeft = tuple(c[c[:, :, 0].argmin()][0])
          extRight = tuple(c[c[:, :, 0].argmax()][0])
          extTop = tuple(c[c[:, :, 1].argmin()][0])
          extBot = tuple(c[c[:, :, 1].argmax()][0])

          ADD_PIXELS = add_pixels_value
          new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
              extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
          set_new.append(new_img)

          return np.array(set_new)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/braintumor')
def brain_tumor():
   return render_template('braintumor.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')

@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')

@app.route("/predict") #default route
def test():
    return render_template("predict.html")#rendering html page

@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetespedigree = request.form['diabetespedigree']
        age = request.form['age']
        skinthickness = request.form['skin']
        pred = diabetes_model.predict(
            [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]])
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultd.html', fn=firstname, ln=lastname, age=int(age), r=pred, gender=gender, glu=int(glucose))
    

@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        pred = heart_model.predict(
            np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender, mhra=mhra)

@app.route('/resultbt', methods=['POST'])
def resultbt():
    if request.method == 'POST':
         firstname = request.form['firstname']
         lastname = request.form['lastname']
         email = request.form['email']
         phone = request.form['phone']
         gender = request.form['gender']
         age = request.form['age']
         headaches = request.form['headaches']
         vision= request.form['vision']
         neurology= request.form['neurology']
         if neurology=="Yes" and vision=="Yes" and headaches=="Yes":
             
            file = request.files['file']
            if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    flash('Image successfully uploaded and displayed below')
                    img = cv2.imread('static/uploads/'+filename)
                    img = crop_imgs([img])
                    img = img.reshape(img.shape[1:])
                    img = preprocess_imgs([img], (224, 224))
                    pred = braintumor_model.predict(img)
                    if pred < 0.5:
                        pred = 0
                    else:
                        pred = 1
            
             # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Brain Tumor test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
            return render_template('resultbt.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender,headaches=headaches,vision=vision,neurology=neurology)
         else:
                pred=0
                return render_template('resultbt.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender,headaches=headaches,vision=vision,neurology=neurology)
        
     
    else:
                flash('Allowed image types are - png, jpg, jpeg')
                return redirect(request.url)

         

@app.route('/resulta', methods=['GET','POST'])        
def resulta():
    if request.method == 'POST':
          print(request.url)
          firstname = request.form['firstname']
          lastname = request.form['lastname']
          email = request.form['email']
          phone = request.form['phone']
          gender = request.form['gender']
          age = request.form['age']
          memory=request.form['memory']
          cognitive=request.form['cognitive']
          if memory=='Yes' and cognitive=='Yes':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                flash('Image successfully uploaded and displayed below')
                img = cv2.imread('static/uploads/'+filename)
                img = cv2.resize(img, (176, 176))
                img = img.reshape(1, 176, 176, 3)
                img = img/255.0
                pred = alzheimer_model.predict(img)
                pred = pred[0].argmax()
                print(pred)
#             # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Alzheimer test results are ready.\nRESULT: {}'.format(firstname,['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'][pred]))
            return render_template('resulta.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender,memory=memory,cognitive=cognitive)
          else:
               pred=-1
               return render_template('resulta.html',  fn=firstname, ln=lastname, age=age, r=pred, gender=gender,memory=memory,cognitive=cognitive)

    else:
             flash('Allowed image types are - png, jpg, jpeg')
             return redirect('/')



@app.route("/call_python_function",methods=["GET","POST"]) #route for our prediction
def call_python_function():
    if request.method=='POST':
        f=request.files['file'] #requesting the file
        basepath=os.path.dirname('__file__')#storing the file directory
        filepath=os.path.join(basepath,"templates",f.filename)#storing the file in uploads folder
        f.save(filepath)#saving the file
        
        img=image.load_img(filepath,target_size=(64,64)) #load and reshaping the image
        x=image.img_to_array(img)#converting image to array
        x=np.expand_dims(x,axis=0)#changing the dimensions of the image
        
        pred=model.predict(x)#predicting classes
        y_pred = np.argmax(pred)
        print("prediction",y_pred)#printing the prediction
    
        index=['Left Bundle Branch Block','Normal','Premature Atrial Contraction',
       'Premature Ventricular Contractions', 'Right Bundle Branch Block','Ventricular Fibrillation']
        result=str(index[y_pred])

        return result#resturing the result
    return None

if __name__ == '__main__':
    app.run(debug=True)
