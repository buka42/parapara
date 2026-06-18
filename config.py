"""
Wspólna konfiguracja aplikacji "Półautomatyczny asystent wiedzy".

Trzymanie wszystkich parametrów w jednym miejscu (modele, ścieżki, długość
bufora itd.) pozwala dostrajać działanie bez grzebania w logice ingest.py
i main.py.
"""
from pathlib import Path

# --- Ścieżki ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
NOTES_DIR = BASE_DIR / "notatki"        # folder z notatkami (.txt, .md, .pdf)
INDEX_DIR = BASE_DIR / "faiss_index"    # tu zapisywany jest indeks wektorowy

# --- Audio / nasłuch -------------------------------------------------------
SAMPLE_RATE = 16000        # faster-whisper pracuje na 16 kHz mono
CHANNELS = 1
AUDIO_DTYPE = "float32"    # sounddevice odda od razu float32 w zakresie [-1, 1]
BUFFER_SECONDS = 30        # "rolling buffer" – ile sekund mowy trzymamy w pamięci
CHUNK_SECONDS = 5          # co ile sekund audio robimy jedną transkrypcję
BLOCKSIZE = 4000           # rozmiar bloku z sounddevice (~0.25 s przy 16 kHz)
AUDIO_QUEUE_MAXSIZE = 256  # zabezpieczenie pamięci, gdy transkrypcja nie nadąża

# --- faster-whisper --------------------------------------------------------
# Modele: tiny, base, small, medium, large-v3. Większy = dokładniej, ale wolniej.
WHISPER_MODEL = "small"
WHISPER_DEVICE = "cpu"     # "cuda" przy karcie NVIDIA z CUDA
WHISPER_COMPUTE = "int8"   # "int8" (CPU), "float16" (GPU)
WHISPER_LANGUAGE = "pl"    # ustaw None dla automatycznego wykrywania języka

# --- Ollama (LLM lokalny) --------------------------------------------------
OLLAMA_MODEL = "llama3"                  # ekstrakcja pytania + generowanie odpowiedzi
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # model embeddingów do RAG
OLLAMA_TEMPERATURE = 0.2                 # nisko = mniej "fantazjowania"

# --- RAG -------------------------------------------------------------------
CHUNK_SIZE = 1000          # długość fragmentu tekstu (znaki) przy chunkowaniu
CHUNK_OVERLAP = 150        # zakładka między fragmentami (zachowanie kontekstu)
RETRIEVER_K = 4            # ile fragmentów notatek trafia do kontekstu odpowiedzi

# --- Skrót klawiszowy ------------------------------------------------------
HOTKEY = "f12"
