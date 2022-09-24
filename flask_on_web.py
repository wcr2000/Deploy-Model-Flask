from flask import Flask,render_template,request
import joblib
import pandas as pd 

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():

    vectorizer = joblib.load(open("vectorizer.pkl",'rb'))
    classifier = joblib.load(open('LR_model.pkl','rb'))

    if request.method == 'POST':
        message = request.form['message']
        unknown = pd.DataFrame(
                {'sentence': [
                message
            ]}
        )
        unknown_vectors = vectorizer.transform(unknown.sentence).toarray()
        my_prediction = classifier.predict(unknown_vectors)
        
    return render_template('home.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)