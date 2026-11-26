import os
from llama_cpp import Llama

# --------------------------------------------------
# Ram Configuration
# --------------------------------------------------

RAM_MODEL_PATH = os.getenv("RAM_GGUF", "models/mistral-q2_k.gguf")

MODEL_CONFIG = {
    "n_ctx": 4096,
    "n_threads": 8,
    "n_gpu_layers": 20
}

GENERATION_CONFIG = {
    "max_tokens": 160,
    "temperature": 0.75,
    "top_p": 0.9,
    "repeat_penalty": 1.2,
    "stop": ["Ram:", "Krishna:", "User:"],
    "echo": False
}

SUMMARY_CONFIG = {
    "max_tokens": 400,
    "temperature": 0.5,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "stop": ["User:", "Ram:", "Krishna:"],
    "echo": False
}

# --------------------------------------------------
# Ram Unified Identity
# --------------------------------------------------

RAM_IDENTITY = """
You are Ram, a risk-aware systems analyst with a naturally cautious and realistic mindset.

You tend to think about constraints, tradeoffs, and what could go wrong — but you are not robotic or aggressive about it. It is simply how you see the world.

When chatting with a user:
- Respond naturally and directly to whatever they say
- Be friendly, thoughtful, and genuine
- Only bring up risks or constraints if it is actually relevant to what they asked
- If someone says "hi", just say hi back like a normal person

When debating Krishna on a topic:
- Engage naturally with what Krishna just said
- Challenge overoptimistic claims with grounded reasoning
- React to the conversation as it evolves — don't repeat points already made
- Build toward a conclusion as the debate matures
- Be direct but not combative

CRITICAL RULES:
- NEVER cite statistics, studies, or made-up numbers
- If uncertain, phrase it as a question or concern
- Stay concise (2-4 sentences)
"""

# --------------------------------------------------
# Load Ram Model
# --------------------------------------------------

print("Loading Ram (Mistral)...")
ram_llm = Llama(
    model_path=RAM_MODEL_PATH,
    **MODEL_CONFIG
)
print("Ram loaded.\n")

# --------------------------------------------------
# Generate Response
# --------------------------------------------------

def generate_ram_response(prompt, generation_config=None):
    if generation_config is None:
        generation_config = GENERATION_CONFIG
    output = ram_llm(prompt, **generation_config)
    return output["choices"][0]["text"].strip()

# --------------------------------------------------
# Chat Handler
# --------------------------------------------------

def handle_ram_chat(message, conversation_history=None):
    context = ""
    if conversation_history:
        for msg in conversation_history[-5:]:
            role = msg.get("role", "")
            text = msg.get("text", "")
            if role == "user":
                context += f"User: {text}\n"
            elif role == "ram":
                context += f"Ram: {text}\n"

    prompt = f"""
{RAM_IDENTITY}

Situation: You are having a direct conversation with a user.

Previous conversation:
{context if context else "None"}

User: {message}

Ram:"""

    return generate_ram_response(prompt)