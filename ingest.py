"""
ingest.py — Moduł A: przygotowanie bazy wiedzy (RAG).

Ładuje pliki .txt / .md / .pdf z folderu `notatki/`, dzieli ich treść na
fragmenty (chunking) i buduje lokalny indeks wektorowy FAISS przy użyciu
embeddingów OpenAI. Indeks zapisywany jest na dysk (`faiss_index/`) i
wczytywany później przez main.py.

Wymaga zmiennej środowiskowej OPENAI_API_KEY.

Uruchomienie:
    python ingest.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import config

# Importy bibliotek opakowane w czytelny komunikat o brakujących zależnościach.
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:  # pragma: no cover
    print(
        "BŁĄD: brakuje zależności. Zainstaluj je poleceniem:\n"
        "    pip install -r requirements.txt\n"
        f"Szczegóły: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

# Obsługiwane rozszerzenia plików tekstowych (ładowane jako zwykły tekst).
TEXT_SUFFIXES = {".txt", ".md"}


def load_documents(notes_dir: Path) -> list[Document]:
    """Wczytuje wszystkie obsługiwane pliki z folderu notatek (rekurencyjnie).

    Błąd pojedynczego pliku jest raportowany, ale NIE przerywa całego procesu —
    pozostałe pliki zostaną wczytane.
    """
    if not notes_dir.exists():
        raise FileNotFoundError(
            f"Folder z notatkami nie istnieje: {notes_dir}\n"
            "Utwórz go i wrzuć tam pliki .txt / .md / .pdf."
        )

    files = sorted(p for p in notes_dir.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"Folder {notes_dir} jest pusty — brak notatek do wczytania."
        )

    documents: list[Document] = []
    for path in files:
        suffix = path.suffix.lower()
        try:
            if suffix in TEXT_SUFFIXES:
                # autodetect_encoding ratuje pliki zapisane w innym kodowaniu.
                loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
            elif suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                print(f"  - pomijam (nieobsługiwany format): {path.name}")
                continue

            loaded = loader.load()
            documents.extend(loaded)
            print(f"  + wczytano: {path.name} ({len(loaded)} fragm. źródłowych)")
        except Exception as exc:  # noqa: BLE001 - chcemy kontynuować mimo błędu pliku
            print(f"  ! błąd wczytywania {path.name}: {exc}", file=sys.stderr)

    if not documents:
        raise ValueError(
            "Nie udało się wczytać żadnego dokumentu. Sprawdź zawartość folderu "
            f"{notes_dir}."
        )
    return documents


def build_index() -> None:
    """Pełny przebieg: wczytanie -> chunking -> embeddingi -> zapis indeksu."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "Brak zmiennej środowiskowej OPENAI_API_KEY. Ustaw ją przed uruchomieniem, np.:\n"
            "    export OPENAI_API_KEY=sk-...        (Linux/macOS)\n"
            "    setx OPENAI_API_KEY sk-...          (Windows)"
        )

    print(f"[1/4] Wczytywanie notatek z: {config.NOTES_DIR}")
    documents = load_documents(config.NOTES_DIR)
    print(f"      Łącznie wczytano {len(documents)} dokumentów źródłowych.")

    print("[2/4] Dzielenie tekstu na fragmenty (chunking)…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # Tniemy najpierw po akapitach, potem zdaniach, na końcu po znakach.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("Po podziale nie powstały żadne fragmenty — sprawdź notatki.")
    print(f"      Powstało {len(chunks)} fragmentów.")

    print(f"[3/4] Liczenie embeddingów modelem OpenAI '{config.OPENAI_EMBED_MODEL}'…")
    embeddings = config.build_embeddings()
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Nie udało się policzyć embeddingów. Sprawdź, czy klucz OPENAI_API_KEY "
            "jest poprawny i czy masz dostęp do internetu.\n"
            f"Szczegóły: {exc}"
        ) from exc

    print(f"[4/4] Zapisywanie indeksu FAISS do: {config.INDEX_DIR}")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(config.INDEX_DIR))

    print("\nGotowe! Baza wiedzy została zbudowana.")
    print("Możesz teraz uruchomić aplikację:  python main.py")


def main() -> int:
    try:
        build_index()
        return 0
    except KeyboardInterrupt:
        print("\nPrzerwano.")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\nBŁĄD: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
