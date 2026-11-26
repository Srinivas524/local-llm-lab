import threading
import time
import queue
from models.ram import generate_ram_response, RAM_IDENTITY
from models.krishna import generate_krishna_response, KRISHNA_IDENTITY

# --------------------------------------------------
# Debate Configuration
# --------------------------------------------------

DEBATE_DELAY = 6

# --------------------------------------------------
# Debate State
# --------------------------------------------------

debate_topic = ""
debate_history = []
debate_active = False
debate_paused = False
debate_thread = None
output_queue = queue.Queue()
current_turn = "ram"
turn_count = 0
max_turns = 10
debate_summary = None

lock = threading.Lock()

# --------------------------------------------------
# Generate Debate Response
# --------------------------------------------------

def generate_debate_response(identity, generator, speaker_label, opponent_label):
    global debate_history, turn_count

    recent_history = "\n".join(debate_history[-6:]) if debate_history else "The debate is just beginning."

    prompt = f"""
{identity}

Situation: You are debating {opponent_label} on the following topic:
{debate_topic}

Conversation so far:
{recent_history}

{speaker_label}:"""

    response = generator(prompt)
    turn_count += 1
    return response

# --------------------------------------------------
# Generate Summary
# --------------------------------------------------

def generate_summary():
    global debate_summary

    full_history = "\n".join(debate_history)

    prompt = f"""Below is a completed debate transcript. Read it and produce a summary.

### DEBATE TOPIC: {debate_topic}

### TRANSCRIPT:
{full_history}

### SUMMARY:
Give exactly 4-5 key takeaways from this debate.
- Each point must start with a dash (-)
- Each point must be under 12 words
- No extra commentary, just the points

-"""

    try:
        from models.ram import generate_ram_response, SUMMARY_CONFIG, ram_llm
        ram_llm.reset()
        debate_summary = generate_ram_response(prompt, generation_config=SUMMARY_CONFIG)
    except ImportError:
        # API pair — retry on rate limit
        from models.ram import generate_ram_response
        for attempt in range(3):
            try:
                debate_summary = generate_ram_response(prompt)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(15)
                else:
                    print(f"Summary generation failed: {e}")
                    debate_summary = None

# --------------------------------------------------
# Debate Loop
# --------------------------------------------------

def debate_loop():
    global debate_active, debate_paused, current_turn

    while debate_active:
        if debate_paused:
            time.sleep(1)
            continue

        with lock:
            try:
                if current_turn == "ram":
                    reply = generate_debate_response(
                        RAM_IDENTITY,
                        generate_ram_response,
                        "Ram",
                        "Krishna"
                    )
                    debate_history.append(f"Ram: {reply}")
                    output_queue.put({"role": "ram", "text": reply})
                    current_turn = "krishna"

                else:
                    reply = generate_debate_response(
                        KRISHNA_IDENTITY,
                        generate_krishna_response,
                        "Krishna",
                        "Ram"
                    )
                    debate_history.append(f"Krishna: {reply}")
                    output_queue.put({"role": "krishna", "text": reply})
                    current_turn = "ram"

            except Exception as e:
                output_queue.put({"role": "system", "text": f"[{current_turn} failed to respond: {e}]"})
                time.sleep(DEBATE_DELAY)
                continue

        if turn_count >= max_turns:
            debate_active = False
            generate_summary()
            break

        time.sleep(DEBATE_DELAY)

# --------------------------------------------------
# Control Functions
# --------------------------------------------------

def set_debate_topic(topic):
    global debate_topic, debate_history, turn_count, debate_summary
    debate_topic = topic
    debate_history = []
    turn_count = 0
    debate_summary = None
    with output_queue.mutex:
        output_queue.queue.clear()

def get_debate_topic():
    return debate_topic

def start_debate(turns=10):
    global debate_active, debate_paused, debate_thread, current_turn, turn_count, max_turns, debate_summary
    if not debate_active:
        debate_active = True
        debate_paused = False
        current_turn = "ram"
        turn_count = 0
        max_turns = turns
        debate_summary = None
        with output_queue.mutex:
            output_queue.queue.clear()
        debate_thread = threading.Thread(target=debate_loop)
        debate_thread.start()

def pause_debate():
    global debate_paused
    debate_paused = True

def resume_debate():
    global debate_paused
    debate_paused = False

def stop_debate():
    global debate_active, debate_paused
    debate_active = False
    debate_paused = False
    if debate_history:
        threading.Thread(target=generate_summary).start()

def get_latest_output():
    try:
        return output_queue.get_nowait()
    except queue.Empty:
        return None

def get_summary():
    return debate_summary