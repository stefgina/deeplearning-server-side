from flask import Flask, render_template, url_for, request, redirect, send_file, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.utils import secure_filename
from db import db_init, db
from models import Img
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from tensorflow.keras.preprocessing import image

import tensorflow as tf
import os






app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)

@app.route('/', methods=['POST', 'GET'])
def index():

    images = Img.query.order_by(Img.date_created).all()
    return render_template('form.html', images = images)
    

@app.route('/delete/<int:id>')
def delete(id):
    image_to_delete = Img.query.get_or_404(id)

    try:
        db.session.delete(image_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting this image'

"""
def prepare_image(path):
    img_path = 'img/uploads/' + path
    img = image.load_img(img_path + file, target_size = (224,224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
"""

@app.route('/detect/<int:id>')
def detect(id):

    X = Img.query.filter_by(id=id).first()
    img_path = 'img/uploads/' + X.name
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=[224, 224])

    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis,...])
    labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    labels = np.array(open(labels_path).read().splitlines())
    
    model = tf.keras.applications.MobileNetV2()
    predictions = model(x)

    sortedpred = np.argsort(predictions)
    top_5_classes_index = sortedpred[0 , ::-1][:5]+1
    top_5_classes = labels[top_5_classes_index]


    return "<html> <body>  <div> <h1>" + top_5_classes[0] + "</h1> <br/> <h1>" + top_5_classes[1] + "</h1> <br/> <h1>" + top_5_classes[2] + "</h1> <br/> <h1>" + top_5_classes[3] + "</h1> <br/> <h1>" + top_5_classes[4] + "</h1> </div> </body> </html>"
    # return render_template('form.html', selectedImage = "http://127.0.0.1:5000/get/" + str(id))



"""
@app.route('/get/<int:id>')
def get_img(id):
    img = Img.query.filter_by(id=id).first()
    
    if not img:
        return 'Img Not Found!', 404
    
    return Response(img.img, mimetype=img.mimetype)
"""

app.config["IMAGE_UPLOADS"] = "img/uploads"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPG"]


@app.route("/upload-image", methods=['GET','POST'])
def upload_image():


    if request.method == "POST":
        if request.files:
            image = request.files["image"]

            if not image:
                print("No image uploaded")
                return redirect('/')

            mimetype = image.mimetype

            if not mimetype:
                print("Bad mimetype!")
                return redirect('/')
            
            if image.filename == "":
                print("Image must have a filename")
                return redirect('/')
            
            if not allowed_image(image.filename):
                print("That image extension is not allowed")
                return redirect('/')

            else:
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
                image = Img(name=filename, mimetype=mimetype)
                db.session.add(image)
                db.session.commit()
            
            print("Image saved")
            return redirect('/')

    return render_template("form.html")

def finds(): 
    test_datagen = ImageDataGenerator(rescale = 1./255) 
    vals = ['Cat', 'Dog'] # change this according to what you've trained your model to do 
    test_dir = 'uploaded'
    test_generator = test_datagen.flow_from_directory( 
            test_dir, 
            target_size =(224, 224), 
            color_mode ="rgb", 
            shuffle = False, 
            class_mode ='categorical', 
            batch_size = 1) 
  
    pred = model.predict_generator(test_generator) 
    print(pred) 
    return str(vals[np.argmax(pred)]) 

def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

if __name__=="__main__":
    app.run(debug=True)

