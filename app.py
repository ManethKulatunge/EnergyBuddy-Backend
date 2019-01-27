#app.py

from flask import Flask, request, jsonify, Response #import main Flask class and request object
from flask_cors import CORS
import json
import pandas as pd

app = Flask(__name__) #create the Flask app

output_data = pd.read_csv('final_table.csv')
CORS(app)


@app.route('/result')
def data():
    city = request.args.get('city')
    result = output_data.loc[output_data['City'] == city.title()]
    d = {}
    for index, data in result.iterrows():
        d = data.to_dict()
    #return (json.dumps(result), mimetype='application/json')
    return jsonify(d)

if __name__ == '__main__':
    app.run(host="172.30.151.182", debug=True, port=5000) #run app in debug mode on port 5000
 