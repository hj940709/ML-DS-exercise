from flask import Flask
from flask import request
from scipy import misc
import json
app = Flask(__name__,static_folder="./")
@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)
@app.route('/')
def Index():
    return app.send_static_file("Index.html")

@app.route('/image', methods=['POST'])
def doPost():
    try:
        return analysis(misc.imread(request.files['image']))
    except Exception as e:
        print(e)
        return json.dumps({"status":"Invalid Input"})

def analysis(data):
    result={"status":"Error"}
    try:
        result["prob"]=[{"class1":0.01,"class2":0.01}]
        result["status"]="Success"
    except Exception as e:
        print(e)

    return json.dumps(result)






if __name__ == "__main__":
    app.run()