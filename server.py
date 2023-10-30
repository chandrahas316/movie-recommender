import os
from flask import Flask, render_template,url_for
from flask import request
from query import query_processing
import pandas as pd
import time
import pickle
import ast

app = Flask(__name__)
app.config["DEBUG"]= True


@app.route("/")
def index():
    '''
    Renders the starting query page.
    '''
    return render_template("index.html",docs=[]);   


@app.route('/search' , methods=['POST'])
def search():
    print("Time required for querying")
    start_time = time.time()
    ranks= query_processing(request.form['query'])
    file = open("movie_data.obj",'rb')
    df = pickle.load(file)
    file.close()
    docs=[]
    for i in range(0,10) :
        docs+=[df.iloc[ranks[i][1]].to_dict()]
    docs = [[values for keys, values in docs[x].items() ]for x in range(0,10) ]
    print("--- %s seconds ---" % (time.time() - start_time))
    return render_template("index.html",docs=docs);  
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)

# with open('./static/js/metrics.js') as dataFile:
#     data = dataFile.read()
#     str = data[data.find('[') : data.rfind(']')+1]
#     li = ast.literal_eval(str)

# print(li)