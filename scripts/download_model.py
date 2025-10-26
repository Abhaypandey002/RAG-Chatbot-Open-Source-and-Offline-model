# Downloads a CPU-friendly Mistral-7B-Instruct GGUF for llama-cpp (Windows OK)
import requests
from pathlib import Path

# MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
# URL = f"https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/{MODEL}"

# ✅ fixed name
MODEL = "mistral-7b-instruct-v0.2.Q2_K.gguf"
URL = f"https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/{MODEL}"


def main():
    models = Path("models"); models.mkdir(parents=True, exist_ok=True)
    dest = models / MODEL
    if dest.exists():
        print(f"Model already exists: {dest}")
        return
    print(f"Downloading {MODEL} ...")
    with requests.get(URL, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)
    print(f"Saved: {dest}")

if __name__ == "__main__":
    main()
