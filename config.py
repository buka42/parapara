"""
Wspólna konfiguracja aplikacji "Półautomatyczny asystent wiedzy" (wersja chmurowa).

Architektura:
  * LLM (wyodrębnienie pytania + odpowiedź) ...... Claude (Anthropic API)
  * transkrypcja mowy ............................ OpenAI Whisper (chmura)
  * embeddingi do RAG ........................... OpenAI (chmura)
  * baza wektorowa .............................. FAISS (lokalnie)

Słaby komputer jedynie przechwytuje dźwięk z mikrofonu — całe "ciężkie"
przetwarzanie odbywa się w chmurze.
"""
from pathlib import Path

# --- Ścieżki ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
NOTES_DIR = BASE_DIR / "notatki"        # folder z notatkami (.txt, .md, .pdf)
INDEX_DIR = BASE_DIR / "faiss_index"    # tu zapisywany jest indeks wektorowy

# --- Klucze API (.env) -----------------------------------------------------
# Wczytujemy plik `.env` z katalogu projektu, dzięki czemu kluczy
# (ANTHROPIC_API_KEY, OPENAI_API_KEY) nie trzeba eksportować przy każdym
# uruchomieniu. python-dotenv jest opcjonalne — bez niego nadal działają
# zwykłe zmienne środowiskowe (mają one pierwszeństwo nad plikiem .env).
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:  # pragma: no cover
    pass

# --- Audio / nasłuch -------------------------------------------------------
# Trzymamy "rolling buffer" SUROWEGO dźwięku z ostatnich BUFFER_SECONDS sekund
# i transkrybujemy go dopiero po wciśnięciu F12 (jeden request do chmury).
SAMPLE_RATE = 16000        # Whisper pracuje na 16 kHz mono
CHANNELS = 1
AUDIO_DTYPE = "float32"    # sounddevice odda od razu float32 w zakresie [-1, 1]
BUFFER_SECONDS = 30        # ile ostatnich sekund dźwięku analizujemy po F12
BLOCKSIZE = 4000           # rozmiar bloku z sounddevice (~0.25 s przy 16 kHz)

# --- Claude (LLM) ----------------------------------------------------------
# Wymaga zmiennej środowiskowej ANTHROPIC_API_KEY.
ANTHROPIC_MODEL = "claude-haiku-4-5"      # najszybszy/najtańszy — idealny na żywo
ANTHROPIC_MAX_TOKENS_QUESTION = 256       # ekstrakcja pytania jest krótka
ANTHROPIC_MAX_TOKENS_ANSWER = 1024        # zwięzła odpowiedź

# --- OpenAI (transkrypcja + embeddingi) ------------------------------------
# Wymaga zmiennej środowiskowej OPENAI_API_KEY.
OPENAI_TRANSCRIBE_MODEL = "whisper-1"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
TRANSCRIBE_LANGUAGE = "pl"   # ISO-639-1; ustaw None dla auto-detekcji języka

# --- RAG -------------------------------------------------------------------
CHUNK_SIZE = 1000          # długość fragmentu tekstu (znaki) przy chunkowaniu
CHUNK_OVERLAP = 150        # zakładka między fragmentami (zachowanie kontekstu)
RETRIEVER_K = 4            # ile fragmentów notatek trafia do kontekstu odpowiedzi

# --- Skrót klawiszowy ------------------------------------------------------
HOTKEY = "f12"


def build_embeddings():
    """Tworzy obiekt embeddingów (OpenAI) używany przez ingest.py i main.py.

    Import jest leniwy, dzięki czemu samo wczytanie config.py pozostaje lekkie.
    Obiekt czyta klucz z OPENAI_API_KEY.
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
