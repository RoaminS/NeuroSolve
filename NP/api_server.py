'''
➡️ Lance le serveur avec :
python api_server.py

➡️ Puis dans un autre terminal :
python ns018_api_push.py
'''

# api_server.py
from flask import Flask, request

app = Flask(__name__)

@app.route("/receive_prediction", methods=["POST"])
def receive_prediction():
    data = request.get_json()
    print("📥 Prédictions reçues :", len(data["data"]))
    return "Prédictions reçues"

@app.route("/receive_alert", methods=["POST"])
def receive_alert():
    data = request.get_json()
    print("📥 Alertes reçues :", len(data["data"]))
    return "Alertes reçues"


@app.route('/upload_session', methods=['POST'])
def upload_session():
    file = request.files['file']
    filename = file.filename
    file.save(f"./uploaded_sessions/{filename}")
    return {"status": "received", "filename": filename}


if __name__ == "__main__":
    app.run(debug=True, port=6000)
