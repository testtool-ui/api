from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Hugging Face Inference API
API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

def query_huggingface(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    try:
        return response.json()
    except ValueError:
        return {"error": "Invalid response from the model."}

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    payload = {"inputs": user_message}
    response = query_huggingface(payload)

    chatbot_reply = response.get("error", "No response") if isinstance(response, dict) else response[0]["generated_text"]
    return jsonify({"reply": chatbot_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
