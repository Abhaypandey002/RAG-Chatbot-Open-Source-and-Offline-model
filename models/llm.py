# import os
# from llama_cpp import Llama

# _llm = None

# def get_llm():
#     global _llm
#     if _llm is None:
#         model_path = os.getenv("GGUF_MODEL_PATH")
#         if not model_path:
#             raise ValueError("GGUF_MODEL_PATH not set in .env")
#         ctx = int(os.getenv("LLAMA_CTX_TOKENS", 4096))
#         gpu_layers = int(os.getenv("LLAMA_GPU_LAYERS", 0))
#         _llm = Llama(model_path=model_path, n_ctx=ctx, n_gpu_layers=gpu_layers, verbose=False)
#     return _llm

# SYSTEM_PROMPT = (
#     "You are a helpful company knowledge assistant. "
#     "Answer ONLY from the provided context. If the context is insufficient, say: "
#     "'Sorry, I couldn't find anything related to your query.'"
# )

# def build_prompt(query: str, contexts: list[str]) -> str:
#     joined = "\n\n".join([f"[Source {i+1}]\n" + c for i, c in enumerate(contexts)])
#     user = (
#         "Use ONLY the sources above to answer. Quote minimal text and cite [Source #] numbers.\n"
#         f"Question: {query}\n"
#     )
#     # Mistral-style prompt
#     prompt = f"[INST] <<SYS>>{SYSTEM_PROMPT}<</SYS>>\n\n{joined}\n\n{user} [/INST]"

#     return prompt

# def generate_answer(prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
#     llm = get_llm()
#     max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", 512))
#     temperature = temperature or float(os.getenv("LLM_TEMPERATURE", 0.2))
#     out = llm(
#         prompt,
#         max_tokens=max_tokens,
#         temperature=temperature,
#         stop=["</s>", "[INST]"],
#     )
#     return out["choices"][0]["text"].strip()




import os
from llama_cpp import Llama

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        model_path = os.getenv("GGUF_MODEL_PATH")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GGUF_MODEL_PATH is '{model_path}', but that file does not exist. "
                f"Put your .gguf in ./models and set GGUF_MODEL_PATH accordingly."
            )
        ctx = int(os.getenv("LLAMA_CTX_TOKENS", 4096))
        gpu_layers = int(os.getenv("LLAMA_GPU_LAYERS", 0))  # keep 0 for CPU-only
        _llm = Llama(model_path=model_path, n_ctx=ctx, n_gpu_layers=gpu_layers, verbose=False)
    return _llm

SYSTEM_PROMPT = (
    "You are a helpful knowledge assistant. "
    "Use the provided conversation context and sources. "
    "If the sources do not contain the answer, say: "
    "'Sorry, I couldn't find anything related to your query.' "
    "Prefer concise answers and cite [Source #] numbers for claims derived from sources."
)

def build_prompt(query: str, contexts: list[str], chat_context: str | None = None) -> str:
    """
    Builds a Mistral-style [INST] prompt. We include:
    - optional conversation memory (chat_context)
    - retrieved source chunks (contexts)
    - the user's standalone question
    """
    parts = []
    if chat_context:
        parts.append(f"Conversation (last turns):\n{chat_context}")
    if contexts:
        joined_ctx = "\n\n".join([f"[Source {i+1}]\n{c}" for i, c in enumerate(contexts)])
        parts.append(joined_ctx)
    parts.append(f"Question: {query}")
    body = "\n\n".join(parts)

    # NOTE: no leading "<s>" to avoid duplicate warning
    prompt = f"[INST] <<SYS>>{SYSTEM_PROMPT}<</SYS>>\n\n{body}\n[/INST]"
    return prompt

def generate_answer(prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
    llm = get_llm()
    max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", 512))
    temperature = temperature or float(os.getenv("LLM_TEMPERATURE", 0.2))
    out = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["</s>", "[INST]"],
    )
    return out["choices"][0]["text"].strip()
