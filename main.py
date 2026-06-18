"""
main.py — Moduł B: aplikacja główna "Półautomatyczny asystent wiedzy".

Co robi:
  * w tle nasłuchuje mikrofonu (sounddevice) i ciągle transkrybuje mowę
    modelem faster-whisper, utrzymując "rolling buffer" tekstu z ostatnich
    ~30 sekund (starszy tekst jest automatycznie usuwany),
  * po wciśnięciu globalnego skrótu F12:
        1. pobiera bieżącą zawartość bufora,
        2. prosi lokalny model w Ollamie o wyodrębnienie pytania,
        3. przeszukuje notatki (RAG / FAISS),
        4. generuje zwięzłą odpowiedź WYŁĄCZNIE na bazie znalezionych
           fragmentów,
        5. pokazuje wynik w pływającym oknie (always-on-top).

Model wątków (kluczowy, by GUI się nie zacinało — szczegóły w README.md):
  * wątek główny ............ pętla zdarzeń PyQt6 (GUI),
  * wątek PortAudio (callback) wrzuca próbki audio do kolejki,
  * wątek transkrypcji ...... faster-whisper -> rolling buffer tekstu,
  * wątek biblioteki keyboard uruchamia wątek "pipeline" po wciśnięciu F12,
  * wątek pipeline .......... wolne operacje LLM/RAG; wynik do GUI WYŁĄCZNIE
                             przez sygnały Qt (bezpieczne między wątkami).

Uruchomienie:
    python main.py
(wymaga wcześniejszego `python ingest.py` oraz działającego serwera Ollama)
"""
from __future__ import annotations

import queue
import signal
import sys
import threading
import time
from collections import deque

import numpy as np

import config

# Importy zewnętrzne opakowane w czytelny komunikat o brakujących zależnościach.
try:
    import keyboard
    import sounddevice as sd
    from faster_whisper import WhisperModel
    from langchain_community.vectorstores import FAISS
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    print(
        "BŁĄD: brakuje zależności. Zainstaluj je poleceniem:\n"
        "    pip install -r requirements.txt\n"
        f"Szczegóły: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Prompty systemowe dla modelu LLM
# ---------------------------------------------------------------------------
EXTRACT_SYSTEM_PROMPT = (
    "Jesteś asystentem, który z transkrypcji mowy wyodrębnia główne pytanie. "
    "Z poniższego tekstu wyekstrahuj najważniejsze pytanie. Zignoruj dygresje, "
    "powitania i wtrącenia. Zwróć WYŁĄCZNIE treść pytania, bez komentarzy ani "
    "cudzysłowów. Jeśli w tekście nie ma wyraźnego pytania, sformułuj jedno "
    "najbardziej prawdopodobne pytanie na podstawie poruszanego tematu."
)

ANSWER_SYSTEM_PROMPT = (
    "Jesteś precyzyjnym asystentem wiedzy. Odpowiadasz po polsku — krótko i "
    "zwięźle. Odpowiedz na pytanie WYŁĄCZNIE na podstawie podanych fragmentów "
    "notatek. Nie korzystaj z wiedzy spoza notatek. Jeśli w notatkach nie ma "
    "odpowiedzi, napisz dokładnie: 'Brak informacji w notatkach.'"
)

# Styl pływającego okna (ciemne, półprzezroczyste, zaokrąglone).
STYLE_SHEET = """
#root {
    background-color: rgba(20, 22, 28, 235);
    border: 1px solid rgba(120, 130, 150, 120);
    border-radius: 12px;
}
QLabel { color: #e8eaed; }
#title   { font-weight: 600; color: #9ecbff; }
#status  { color: #9aa0a6; font-size: 11px; }
#question{ color: #ffd27f; font-size: 13px; font-weight: 600; }
#answer  { color: #e8eaed; font-size: 14px; }
#sources { color: #7f868d; font-size: 10px; }
#scroll  { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
#close {
    color: #e8eaed; background: rgba(255, 255, 255, 20);
    border: none; border-radius: 11px; font-weight: 700;
}
#close:hover { background: rgba(255, 90, 90, 160); }
"""


# ---------------------------------------------------------------------------
# Audio: ciągły nasłuch mikrofonu -> kolejka próbek
# ---------------------------------------------------------------------------
class AudioListener:
    """Nasłuchuje mikrofonu i wrzuca bloki próbek (float32, mono) do kolejki.

    Callback sounddevice działa na osobnym wątku PortAudio, dlatego MUSI być
    krótki i nieblokujący — całe "ciężkie" przetwarzanie odbywa się w wątku
    transkrypcji.
    """

    def __init__(self, audio_queue: "queue.Queue[np.ndarray]") -> None:
        self._queue = audio_queue
        self._stream: "sd.InputStream | None" = None

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype=config.AUDIO_DTYPE,
            blocksize=config.BLOCKSIZE,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # Np. przepełnienia bufora — tylko logujemy, nie przerywamy nasłuchu.
            print(f"[audio] status: {status}", file=sys.stderr)
        # Bierzemy pierwszy (jedyny) kanał i KOPIUJEMY — bufor jest reużywany.
        block = indata[:, 0].copy()
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            # Transkrypcja nie nadąża: usuwamy najstarszy blok, by trzymać się
            # "na żywo" zamiast budować rosnące opóźnienie.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(block)
            except queue.Empty:
                pass

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


# ---------------------------------------------------------------------------
# Transkrypcja: kolejka próbek -> rolling buffer tekstu
# ---------------------------------------------------------------------------
class RollingTranscriber(threading.Thread):
    """Wątek roboczy: pobiera próbki z kolejki, transkrybuje je w paczkach po
    CHUNK_SECONDS i utrzymuje rolling buffer tekstu z ostatnich BUFFER_SECONDS.

    Paczki nie nakładają się na siebie, więc tekst nie duplikuje się w buforze.
    """

    def __init__(self, model: WhisperModel, audio_queue: "queue.Queue[np.ndarray]") -> None:
        super().__init__(name="RollingTranscriber", daemon=True)
        self._model = model
        self._queue = audio_queue
        self._buffer: "deque[tuple[float, str]]" = deque()
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()

    def run(self) -> None:
        samples_per_chunk = int(config.CHUNK_SECONDS * config.SAMPLE_RATE)
        pending: list[np.ndarray] = []
        collected = 0

        while self._running.is_set():
            try:
                block = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue

            pending.append(block)
            collected += len(block)
            if collected < samples_per_chunk:
                continue

            audio = np.concatenate(pending)
            pending.clear()
            collected = 0

            try:
                text = self._transcribe(audio)
            except Exception as exc:  # noqa: BLE001
                print(f"[whisper] błąd transkrypcji: {exc}", file=sys.stderr)
                continue

            if text:
                with self._lock:
                    self._buffer.append((time.monotonic(), text))
                    self._evict_locked()

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language=config.WHISPER_LANGUAGE,
            vad_filter=True,                  # pomija ciszę -> mniej halucynacji
            beam_size=1,                      # szybciej (wystarcza dla mowy)
            condition_on_previous_text=False,  # paczki są od siebie niezależne
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _evict_locked(self) -> None:
        """Usuwa wpisy starsze niż BUFFER_SECONDS. Wywoływać pod blokadą."""
        cutoff = time.monotonic() - config.BUFFER_SECONDS
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def get_transcript(self) -> str:
        """Zwraca scalony tekst z ostatnich BUFFER_SECONDS sekund."""
        with self._lock:
            self._evict_locked()
            return " ".join(text for _, text in self._buffer).strip()

    def stop(self) -> None:
        self._running.clear()


# ---------------------------------------------------------------------------
# RAG: indeks FAISS + LLM (Ollama)
# ---------------------------------------------------------------------------
class KnowledgeBase:
    """Wczytuje indeks FAISS oraz model LLM; wyodrębnia pytanie i generuje
    odpowiedź na podstawie znalezionych fragmentów notatek."""

    def __init__(self) -> None:
        if not config.INDEX_DIR.exists():
            raise FileNotFoundError(
                f"Nie znaleziono indeksu wiedzy w {config.INDEX_DIR}.\n"
                "Uruchom najpierw:  python ingest.py"
            )
        self._embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBED_MODEL)
        self._vector_store = FAISS.load_local(
            str(config.INDEX_DIR),
            self._embeddings,
            allow_dangerous_deserialization=True,  # indeks budujemy lokalnie
        )
        self._retriever = self._vector_store.as_retriever(
            search_kwargs={"k": config.RETRIEVER_K}
        )
        self._llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            temperature=config.OLLAMA_TEMPERATURE,
        )

    def extract_question(self, transcript: str) -> str:
        """Wyciąga z transkrypcji najważniejsze pytanie."""
        messages = [
            SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=transcript),
        ]
        response = self._llm.invoke(messages)
        return response.content.strip().strip('"').strip()

    def answer(self, question: str) -> "tuple[str, list[str]]":
        """Zwraca (odpowiedź, lista nazw plików źródłowych)."""
        docs = self._retriever.invoke(question)
        if not docs:
            return "Brak informacji w notatkach.", []

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"FRAGMENTY NOTATEK:\n{context}\n\nPYTANIE:\n{question}"
            ),
        ]
        response = self._llm.invoke(messages)

        # Unikalne nazwy źródeł (do pokazania pod odpowiedzią).
        sources: list[str] = []
        for doc in docs:
            raw = str(doc.metadata.get("source", "?"))
            name = raw.replace("\\", "/").split("/")[-1]
            if name not in sources:
                sources.append(name)
        return response.content.strip(), sources


# ---------------------------------------------------------------------------
# Most wątek-roboczy -> GUI (sygnały Qt są bezpieczne między wątkami)
# ---------------------------------------------------------------------------
class UiBridge(QObject):
    """Sygnały emitowane przez wątki robocze; sloty wykonują się w wątku GUI
    (połączenie kolejkowane Qt marshaluje wywołanie do wątku głównego)."""

    status = pyqtSignal(str)
    question = pyqtSignal(str)
    result = pyqtSignal(str, list)  # (odpowiedź, źródła)
    error = pyqtSignal(str)


# ---------------------------------------------------------------------------
# Pływające okno (always-on-top)
# ---------------------------------------------------------------------------
class OverlayWindow(QWidget):
    """Proste, pływające, zawsze-na-wierzchu okno z odpowiedzią."""

    def __init__(self) -> None:
        super().__init__()
        self._drag_pos = None
        self._build_ui()
        self._place_bottom_right()

    def _build_ui(self) -> None:
        self.setWindowTitle("Asystent wiedzy")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.resize(440, 340)

        # Wewnętrzna ramka pozwala zrobić zaokrąglone, półprzezroczyste tło.
        root = QFrame(self)
        root.setObjectName("root")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Nagłówek: tytuł (obszar do przeciągania) + przycisk zamknięcia.
        header = QHBoxLayout()
        title = QLabel(f"🧠 Asystent wiedzy  ·  {config.HOTKEY.upper()} = pytanie")
        title.setObjectName("title")
        close_btn = QPushButton("✕")
        close_btn.setObjectName("close")
        close_btn.setFixedSize(22, 22)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(close_btn)
        layout.addLayout(header)

        self.status_label = QLabel("Inicjalizacja…")
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)

        self.question_label = QLabel("")
        self.question_label.setObjectName("question")
        self.question_label.setWordWrap(True)
        layout.addWidget(self.question_label)

        # Odpowiedź w obszarze przewijanym (na wypadek dłuższego tekstu).
        self.answer_label = QLabel("")
        self.answer_label.setObjectName("answer")
        self.answer_label.setWordWrap(True)
        self.answer_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.answer_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        scroll = QScrollArea()
        scroll.setObjectName("scroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.answer_label)
        layout.addWidget(scroll, stretch=1)

        self.sources_label = QLabel("")
        self.sources_label.setObjectName("sources")
        self.sources_label.setWordWrap(True)
        layout.addWidget(self.sources_label)

        self.setStyleSheet(STYLE_SHEET)

    def _place_bottom_right(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(
            area.right() - self.width() - 20,
            area.bottom() - self.height() - 20,
        )

    # --- Sloty aktualizujące GUI (wywoływane w wątku głównym) ---------------
    def on_status(self, text: str) -> None:
        self.status_label.setText(text)

    def on_question(self, text: str) -> None:
        self.question_label.setText(f"❓ {text}")
        self.answer_label.setText("…")
        self.sources_label.setText("")
        self._surface()

    def on_result(self, answer: str, sources: list) -> None:
        self.status_label.setText(f"Gotowe · {config.HOTKEY.upper()}, by zapytać ponownie")
        self.answer_label.setText(answer)
        self.sources_label.setText(
            "📄 Źródła: " + ", ".join(sources) if sources else ""
        )
        self._surface()

    def on_error(self, text: str) -> None:
        self.status_label.setText("⚠️ " + text)
        self._surface()

    def _surface(self) -> None:
        """Pokazuje okno i wynosi je na wierzch (bez kradzieży fokusu)."""
        if not self.isVisible():
            self.show()
        self.raise_()

    # --- Przeciąganie okna bez systemowej ramki -----------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None

    def keyPressEvent(self, event) -> None:
        # Escape tylko chowa okno — F12 przywoła je z nową odpowiedzią.
        if event.key() == Qt.Key.Key_Escape:
            self.hide()

    def closeEvent(self, event) -> None:
        # Zamknięcie okna (✕ lub menedżer okien) kończy całą aplikację;
        # sprzątanie zasobów wykona slot podłączony do aboutToQuit.
        app = QApplication.instance()
        if app is not None:
            app.quit()
        event.accept()


# ---------------------------------------------------------------------------
# Orkiestracja: spina audio, transkrypcję, skrót F12 i pipeline RAG
# ---------------------------------------------------------------------------
class Assistant:
    def __init__(self, bridge: UiBridge) -> None:
        self.bridge = bridge
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(
            maxsize=config.AUDIO_QUEUE_MAXSIZE
        )
        self.listener: "AudioListener | None" = None
        self.transcriber: "RollingTranscriber | None" = None
        self.kb: "KnowledgeBase | None" = None
        self._busy = threading.Lock()  # gwarantuje jeden pipeline na raz
        self._shut = False

    def initialize(self) -> None:
        """Ciężka inicjalizacja (modele, indeks, nasłuch, skrót).

        Uruchamiana w osobnym wątku, dzięki czemu okno GUI pojawia się od razu,
        a użytkownik widzi postęp w pasku statusu.
        """
        try:
            self.bridge.status.emit(
                "Ładowanie modelu Whisper… (pierwsze uruchomienie pobiera model)"
            )
            model = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE,
            )

            self.bridge.status.emit("Wczytywanie bazy wiedzy (FAISS)…")
            self.kb = KnowledgeBase()

            self.bridge.status.emit("Uruchamianie nasłuchu mikrofonu…")
            self.transcriber = RollingTranscriber(model, self._audio_queue)
            self.transcriber.start()
            self.listener = AudioListener(self._audio_queue)
            self.listener.start()

            self._register_hotkey()

            self.bridge.status.emit(
                f"Nasłuchuję… Wciśnij {config.HOTKEY.upper()}, aby zadać pytanie."
            )
        except Exception as exc:  # noqa: BLE001
            self.bridge.error.emit(f"Błąd startu: {exc}")
            print(f"[init] {exc}", file=sys.stderr)

    def _register_hotkey(self) -> None:
        try:
            keyboard.add_hotkey(config.HOTKEY, self._on_hotkey)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Nie udało się zarejestrować skrótu '{config.HOTKEY}'. "
                "Na Linux uruchom z uprawnieniami root (sudo), a na macOS nadaj "
                f"aplikacji uprawnienia Accessibility. Szczegóły: {exc}"
            ) from exc

    def _on_hotkey(self) -> None:
        # Wywoływane w wątku biblioteki keyboard. Jeśli analiza już trwa —
        # ignorujemy wciśnięcie (nie nakładamy zapytań na siebie).
        if self._busy.locked():
            return
        threading.Thread(
            target=self._run_pipeline, name="Pipeline", daemon=True
        ).start()

    def _run_pipeline(self) -> None:
        """Sekwencja wyzwalacza F12: bufor -> pytanie -> RAG -> odpowiedź.

        Cała ciężka praca dzieje się tutaj, w osobnym wątku — GUI pozostaje
        responsywne, a wyniki trafiają do okna wyłącznie przez sygnały Qt.
        """
        if not self._busy.acquire(blocking=False):
            return
        try:
            if self.transcriber is None or self.kb is None:
                self.bridge.error.emit("Asystent jeszcze się uruchamia…")
                return

            self.bridge.status.emit("Analizuję ostatnie sekundy mowy…")
            transcript = self.transcriber.get_transcript()
            if not transcript:
                self.bridge.error.emit(
                    "Bufor pusty — mów do mikrofonu i spróbuj ponownie."
                )
                return

            self.bridge.status.emit("Wyodrębniam pytanie…")
            question = self.kb.extract_question(transcript)
            if not question:
                self.bridge.error.emit("Nie udało się wyodrębnić pytania.")
                return
            self.bridge.question.emit(question)

            self.bridge.status.emit("Szukam w notatkach i generuję odpowiedź…")
            answer, sources = self.kb.answer(question)
            self.bridge.result.emit(answer, sources)
        except Exception as exc:  # noqa: BLE001
            self.bridge.error.emit(f"Błąd: {exc}")
            print(f"[pipeline] {exc}", file=sys.stderr)
        finally:
            self._busy.release()

    def shutdown(self) -> None:
        """Porządne zatrzymanie wątków i zwolnienie zasobów (idempotentne)."""
        if self._shut:
            return
        self._shut = True
        try:
            keyboard.unhook_all()
        except Exception:  # noqa: BLE001
            pass
        if self.listener is not None:
            self.listener.stop()
        if self.transcriber is not None:
            self.transcriber.stop()
            self.transcriber.join(timeout=2.0)


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    bridge = UiBridge()
    window = OverlayWindow()

    # Sygnały (wątki robocze) -> sloty (wątek GUI). Domyślne, kolejkowane
    # połączenie Qt bezpiecznie przenosi wywołanie do wątku głównego.
    bridge.status.connect(window.on_status)
    bridge.question.connect(window.on_question)
    bridge.result.connect(window.on_result)
    bridge.error.connect(window.on_error)

    window.show()

    assistant = Assistant(bridge)
    app.aboutToQuit.connect(assistant.shutdown)

    # Ciężki start w tle — okno pojawia się natychmiast.
    threading.Thread(target=assistant.initialize, name="Init", daemon=True).start()

    # Pozwól interpreterowi obsłużyć Ctrl+C (PyQt bez tego "połyka" SIGINT).
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    pulse = QTimer()
    pulse.start(200)
    pulse.timeout.connect(lambda: None)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
