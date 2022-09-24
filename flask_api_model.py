from json import load
from flask import Flask,render_template,request
import joblib
import pandas as pd 

app = Flask(__name__)

@app.route('/api_pred')
def query_example():
    # if key doesn't exist, returns None
    # vectorizer = joblib.load(open("vectorizer.pkl",'rb'))
    # classifier = joblib.load(open('LR_model.pkl','rb'))
    vectorizer = joblib.load("vectorizer.pkl")
    classifier = joblib.load("LR_model.pkl")

    message = request.args.get('message')

    unknown = pd.DataFrame(
                {'sentence': [
                message
            ]}
        )
    unknown_vectors = vectorizer.transform(unknown.sentence).toarray()
    my_prediction = classifier.predict(unknown_vectors)

    return '''<h1>The number value is: {}</h1>'''.format(my_prediction)

# @app.route('/')
# def home():
# 	return render_template('home.html')


# @app.route('/api',methods=['POST','GET']) # login no register
# def pred_web():

#     # vectorizer = joblib.load(open("vectorizer.pkl",'rb'))
#     # classifier = joblib.load(open('LR_model.pkl','rb'))

#     if request.method == 'POST':
#         #body = request.get_json()

#         message = request.args.get('message')
#         #print(message)

#         # unknown = pd.DataFrame(
#         #         {'sentence': [
#         #         message
#         #     ]}
#         # )
#         # unknown_vectors = vectorizer.transform(unknown.sentence).toarray()
#         # my_prediction = classifier.predict(unknown_vectors)

#         return {message}

#     return {"Message":"Hello!!"},201

# @app.route('/predict',methods=['POST'])
# def predict():

#     vectorizer = joblib.load(open("vectorizer.pkl",'rb'))
#     classifier = joblib.load(open('LR_model.pkl','rb'))

#     if request.method == 'POST':
#         message = request.form['message']
#         unknown = pd.DataFrame(
#                 {'sentence': [
#                 message
#             ]}
#         )
#         unknown_vectors = vectorizer.transform(unknown.sentence).toarray()
#         my_prediction = classifier.predict(unknown_vectors)
        
#     return render_template('home.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)