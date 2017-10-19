from flask import Flask
from flask import request
from scipy import misc
from keras.models import load_model
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
        return json.dumps({"status":500})

def analysis(data):
    def tranfer(class_indices, pred):
        dist = {}
        for key in class_indices.keys():
            dist[key] = pred[class_indices[key]]
        return dist
    result={"status":400}
    try:
        model = load_model("./predictor.h5")
        class_indices = json.load(open("./class_indices.json","r"))
        pred = model.predict_proba(data.reshape((1,150,150,3))).flatten()
        result["dist"] = tranfer(class_indices,pred)
        result["status"]=200
    except Exception as e:
        print(e)
    return json.dumps(result)

if __name__ == "__main__":
    app.run(port=8080)