from flask import Flask, render_template, request, jsonify
from Model_Notebook import get_recommendations
import pandas as pd

app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    recommendation = []
    if request.method == 'POST':
        user_input = request.form.get('title')
        print(user_input, type(user_input))
        recommendation = get_recommendations(user_input)
        print(recommendation, type(recommendation))
        recommendation.append(recommendation)
    return render_template('index.html', recommendation=recommendation)

@app.route('/api', methods=['GET'])
def api_result():
    d ={}
    user_input = request.args['Query']
    d['Query'] = str(get_recommendations(user_input))
    return jsonify(d)

if __name__== '__main__':
    app.run(debug=True)