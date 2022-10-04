import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('venv/models.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    
    if prediction==0:
        return render_template('index.html',
                               prediction_text='Patient has Low chance of getting Diabetes'.format(prediction),
                               )
    else:
        return render_template('index.html',
                               prediction_text= 'Patient has High chance of getting Diabetes'.format(prediction),
                               )

if __name__ == "__main__":
    app.run(debug=True)