import os
import numpy as np
from keras.utils import load_img, img_to_array 
from tensorflow.keras.models import load_model
from flask import Flask, render_template, url_for, request, jsonify
import base64

diseaseInfo = [
    '''Your heart has 4 chambers. The 2 upper chambers are called atria, and the 2 lower chambers are called ventricles. In a healthy heart, the signal to start your heartbeat begins in the upper right chamber of the heart (right atrium). From there, the signal activates the left atrium and travels to the lower chambers (right and left ventricles) of the heart. As the signal travels along the heart's conduction system, it triggers nearby parts of the heart to contract in a coordinated manner.

Two bundle branches carry the electrical signal through the ventricles to the bottom of the heart and cause the ventricles to beat. These are termed the right bundle and left bundle. In left bundle branch block, there is a problem with the left branch of the electrical conduction system. The electrical signal can't travel down this path the way it normally would. The signal still gets to the left ventricle, but it is slowed down. That's because the signal has to spread from the right bundle branch through the heart muscle and slowly activate the left ventricle. So the left ventricle contracts a little later than it normally would. This can cause an uncoordinated contraction of the heart. As a result, the heart may eject blood less efficiently. For most people, this is not a big problem. But if you have underlying heart failure, left bundle branch block can make it worse.

In some people, a left bundle branch block is present all the time. In others, it happens off and on, depending on the heart rate. Exercise, for example, might bring it on for some people. Some people can even have an incomplete left bundle branch block. This would be a sign that a person may be developing a left bundle branch block.

Left bundle branch block happens more often in older people. It is rare in healthy young people. It usually happens in people who have some type of underlying heart problem.''',
'''Congratulations! Your heart is functioning as expected :)''', '''Premature atrial contractions (PACs) are extra heartbeats that start in the upper chambers of your heart. When the premature, or early, signal tells the heart to contract, there may not be much blood in the heart at that moment. That means there's not much blood to pump out. A pause and a strong beat may follow the extra heartbeat, making it feel like a skipped beat.

A premature atrial contraction can feel like an extra beat when there's more blood in the heart to pump than there is with a skipped beat. Premature atrial contractions are very common in adults but rare in children born without heart problems.''', '''Premature ventricular contractions (PVCs) are a type of irregular heartbeat. They occur when the electrical signal that starts your heartbeat comes from one of your bottom two heart chambers (ventricles). The signal typically starts in the top right chamber (atrium).

PVCs are not always a problem. But if they repeatedly happen for months or years, they can cause a type of cardiomyopathy, or heart muscle weakening. PVCs usually go away with medication or other minimally invasive treatments. PVCs are quite common. Up to 75 per cent of people experience them.''', '''Right bundle branch block is an obstacle in your right bundle branch that makes your heartbeat signal late and out of sync with the left bundle branch, creating an irregular heartbeat.

Electrical signals in your heart act like a pacemaker that controls your heartbeats. This signal starts in the sinoatrial (SA) node, which tells your left and right atria (upper heart chambers) to contract. Next, the signal goes to your atrioventricular (AV) node and bundle branches, which make your left and right ventricles tighten.

Normally, the signal goes down both bundle branches at once and both ventricles are working at the same time. When the signal goes down the right bundle branch a little slower than the left bundle branch, the right ventricle contracts later than the left one. This is what's going on when you have a right bundle branch block, and it happens because the signal has to go around a block in the branch.

Because the ventricles aren't working at exactly the same time, this creates an irregular heartbeat (arrhythmia.)

Right bundle branch block can be complete or incomplete. Unlike complete right bundle branch block, incomplete right bundle branch block doesn't increase your risk of heart attack and death. Right bundle branch block can happen in healthy people. It's more likely to happen in older people.''', '''Ventricular fibrillation (sometimes called v-fib for short) is an arrhythmia, a malfunction of the heart's normal pumping sequence. It is the most common deadly arrhythmia.

When it happens, the lower chambers of your heart quiver or twitch instead of completely expanding and squeezing. This means they aren't pumping blood as they should.'''
    
]

def predictCategoryIndex(filepath):
    img = load_img(filepath, target_size=(224,224,3))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    model = load_model('Potato_ResNet152.h5')
    prediction = model.predict(x)
    index = np.argmax(prediction)
    return index

app = Flask(__name__)
upload_path  = 'static/uploads'
@app.route('/')
def about():
    return render_template('about.html')

@app.route('/about')
def home():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        # print('Request contents: ' + request.form['imageData'])
        image = request.form['imageData']
        image_bytes = base64.b64decode(image.split(',')[1])
        filepath = 'static/uploads/trial_image.png'
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        index = predictCategoryIndex(filepath)
        CATEGORIES = ['Healthy', 'Late Blight', 'Early Blight']
        result = CATEGORIES[index]
        print(result)
        return render_template('results.html', prediction=result, filename='trial_image.png')


        # data = request.get_json()
        # print(data)
        # f = request.files['imageData']
        # filepath = os.path.join(upload_path, f.filename)
        # f.save(filepath)
        # print(f.filename)
        # index = predictCategoryIndex(filepath)
        # CATEGORIES = ['Left Bundle Branch Block', 'Normal', 'Premature Atrial Contraction', 'Premature Ventricular Contraction', 'Right Bundle Branch Block', 'Ventricular Fibrillation']
        # result = CATEGORIES[index]
        # print(result)
        # return render_template('results.html', prediction=result, filename=f.filename, diseaseInfo=diseaseInfo[index])
        # return render_template('about.html')
    return None

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # Check if the POST request contains an image file
    # print('Reached the server!')
    if request.method == 'POST':
        data = request.get_json()
        image = data['image']
        image_bytes = base64.b64decode(image.split(',')[1])
        filepath = 'static/images/trial_image.png'
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        index = predictCategoryIndex(filepath)
        CATEGORIES = ['Healthy', 'Late Blight', 'Early Blight']
        result = CATEGORIES[index]
        print(result)
        return render_template('about.html')
    return None

if __name__ == "__main__":
    app.run(debug=True) 