import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn.linear_model._base
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    temparray=[8,8,8,8,7,7,7,7,8,8,9,9,9,8,9,8,8,9,7,7,7]
    n=len(int_features)

    for i in range(n):
        int_features[i]=int_features[i]/temparray[i]


    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)

    output = round(prediction[0])
    if output==1:
        return render_template('index.html', prediction_text='The person may have cancer')
    else:
        return render_template('index.html', prediction_text='You dont have cancer and you are in good health as of now')



if __name__ == "__main__":
    app.run(debug=True)

