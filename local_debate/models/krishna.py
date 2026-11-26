import os
from llama_cpp import Llama

# --------------------------------------------------
# Krishna Configuration
# --------------------------------------------------

KRISHNA_MODEL_PATH = os.getenv("KRISHNA_GGUF", "models/mistral-q2_k.gguf")

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

# --------------------------------------------------
# Krishna Unified Identity
# --------------------------------------------------

KRISHNA_IDENTITY = """
You are Krishna, an optimization-focused systems analyst with a naturally solution-oriented mindset.

You tend to see pathways forward, engineering solutions, and possibilities — but you are not blindly optimistic. You acknowledge real challenges while focusing on how they can be overcome.

When chatting with a user:
- Respond naturally and directly to whatever they say
- Be friendly, thoughtful, and genuine
- Only bring up solutions or optimizations if it is actually relevant to what they asked
- If someone says "hi", just say hi back like a normal person

When debating Ram on a topic:
- Engage naturally with what Ram just said
- Counter pessimism with constructive, grounded alternatives
- React to the conversation as it evolves — don't repeat points already made
- Build toward a conclusion as the debate matures
- Be direct but not dismissive of concerns

CRITICAL RULES:
- NEVER cite statistics, studies, or made-up numbers
- If uncertain, phrase it as "one approach could be" or "potentially"
- Stay concise (2-4 sentences)
"""

# --------------------------------------------------
# Load Krishna Model
# --------------------------------------------------

print("Loading Krishna (Mistral)...")
krishna_llm = Llama(
    model_path=KRISHNA_MODEL_PATH,
    **MODEL_CONFIG
)
print("Krishna loaded.\n")

# --------------------------------------------------
# Generate Response
# --------------------------------------------------

def generate_krishna_response(prompt, generation_config=None):
    if generation_config is None:
        generation_config = GENERATION_CONFIG
    output = krishna_llm(prompt, **generation_config)
    return output["choices"][0]["text"].strip()

# --------------------------------------------------
# Chat Handler
# --------------------------------------------------

def handle_krishna_chat(message, conversation_history=None):
    context = ""
    if conversation_history:
        for msg in conversation_history[-5:]:
            role = msg.get("role", "")
            text = msg.get("text", "")
            if role == "user":
                context += f"User: {text}\n"
            elif role == "krishna":
                context += f"Krishna: {text}\n"

    prompt = f"""
{KRISHNA_IDENTITY}

Situation: You are having a direct conversation with a user.

Previous conversation:
{context if context else "None"}

User: {message}

Krishna:"""

    return generate_krishna_response(prompt)