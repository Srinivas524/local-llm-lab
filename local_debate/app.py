from flask import Flask, request, jsonify
from flask_cors import CORS
from models.ram import handle_ram_chat
from models.krishna import handle_krishna_chat
from debate_handler import (
    set_debate_topic,
    get_debate_topic,
    start_debate,
    pause_debate,
    resume_debate,
    stop_debate,
    get_latest_output,
    get_summary
)

# --------------------------------------------------
# Server Configuration
# --------------------------------------------------

HOST = "0.0.0.0"
PORT = 5050
DEBUG = False

# --------------------------------------------------
# Flask Setup
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Chat Endpoint
# --------------------------------------------------

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        ai = data.get('ai', 'ram')

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        if ai == 'ram':
            response = handle_ram_chat(message)
        else:
            response = handle_krishna_chat(message)

        return jsonify({'response': response})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --------------------------------------------------
# Debate Control Endpoint
# --------------------------------------------------

@app.route("/debate", methods=["POST"])
def debate_control():
    data = request.get_json()
    action = data.get("action")

    if action == "topic":
        topic = data.get("topic", "")
        set_debate_topic(topic)
        return jsonify({"status": "topic set"})

    if action == "start":
        if not get_debate_topic():
            return jsonify({"error": "No topic set"}), 400
        turns = data.get("turns", 10)
        start_debate(turns=turns)
        return jsonify({"status": "started"})

    if action == "pause":
        pause_debate()
        return jsonify({"status": "paused"})

    if action == "resume":
        resume_debate()
        return jsonify({"status": "resumed"})

    if action == "stop":
        stop_debate()
        return jsonify({"status": "stopped"})

    return jsonify({"error": "Invalid action"}), 400

# --------------------------------------------------
# Debate State Polling Endpoint
# --------------------------------------------------

@app.route("/debate_state", methods=["GET"])
def debate_state():
    output = get_latest_output()

    if output:
        return jsonify({
            "new_message": True,
            "role": output["role"],
            "text": output["text"]
        })

    return jsonify({"new_message": False})

# --------------------------------------------------
# Debate Summary Endpoint
# --------------------------------------------------

@app.route("/debate_summary", methods=["GET"])
def debate_summary():
    summary = get_summary()

    if summary:
        return jsonify({"ready": True, "summary": summary})

    return jsonify({"ready": False})

# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":
    print(f"Ramdos AI Arena running at http://{HOST}:{PORT}")
    print("Modes: Chat (Single AI) | Debate (Ram vs Krishna)")
    app.run(host=HOST, port=PORT, debug=DEBUG)