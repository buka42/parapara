"""
main.py — Moduł B: aplikacja główna "Półautomatyczny asystent wiedzy" (chmura).

Co robi:
  * w tle nasłuchuje mikrofonu (sounddevice) i trzyma w pamięci "rolling buffer"
    SUROWEGO dźwięku z ostatnich ~30 sekund (starsze próbki są usuwane),
  * po wciśnięciu globalnego skrótu F12:
        1. pobiera ostatnie ~30 s dźwięku z bufora,
        2. transkrybuje je w chmurze (OpenAI Whisper),
        3. prosi Claude (Anthropic API) o wyodrębnienie pytania,
        4. przeszukuje notatki (RAG / FAISS, embeddingi OpenAI),
        5. Claude generuje zwięzłą odpowiedź WYŁĄCZNIE na bazie znalezionych
           fragmentów,
        6. wynik pojawia się w pływającym oknie (always-on-top).

Dlaczego transkrypcja "na żądanie", a nie ciągła?
  Transkrypcja jest teraz w chmurze, więc ciągłe transkrybowanie generowałoby
  stały ruch i koszt. Zamiast tego trzymamy tani bufor surowego dźwięku i
  wysyłamy do chmury JEDEN request — dopiero po wciśnięciu F12.

Model wątków (kluczowy, by GUI się nie zacinało — szczegóły w README.md):
  * wątek główny ............ pętla zdarzeń PyQt6 (GUI),
  * wątek PortAudio (callback) dopisuje próbki do rolling buffera (pod blokadą),
  * wątek biblioteki keyboard uruchamia wątek "pipeline" po wciśnięciu F12,
  * wątek pipeline .......... wolne operacje sieciowe (OpenAI + Claude + RAG);
                             wynik do GUI WYŁĄCZNIE przez sygnały Qt.

Wymaga zmiennych środowiskowych: ANTHROPIC_API_KEY oraz OPENAI_API_KEY.
Uruchomienie:
    python main.py
(wymaga wcześniejszego `python ingest.py`)
"""
from __future__ import annotations

import io
import os
import signal
import sys
import threading
import wave
from collections import deque

import numpy as np

import config

# Importy zewnętrzne opakowane w czytelny komunikat o brakujących zależnościach.
try:
    import anthropic
    import keyboard
    import sounddevice as sd
    from openai import OpenAI, OpenAIError
    from langchain_community.vectorstores import FAISS
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
# Prompty systemowe dla modelu LLM (Claude)
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
    "zwięźle, bez wstępów. Odpowiedz na pytanie WYŁĄCZNIE na podstawie podanych "
    "fragmentów notatek. Nie korzystaj z wiedzy spoza notatek. Jeśli w notatkach "
    "nie ma odpowiedzi, napisz dokładnie: 'Brak informacji w notatkach.'"
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


def _wav_bytes_from_float32(audio: np.ndarray, sample_rate: int) -> bytes:
    """Konwertuje próbki float32 [-1, 1] (mono) na bajty pliku WAV (PCM 16-bit)."""
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _message_text(message) -> str:
    """Składa tekst z bloków odpowiedzi Claude (pomija ewentualne bloki myślenia)."""
    return "".join(block.text for block in message.content if block.type == "text")


# ---------------------------------------------------------------------------
# Rolling buffer surowego dźwięku (ostatnie BUFFER_SECONDS sekund)
# ---------------------------------------------------------------------------
class AudioRingBuffer:
    """Bezpieczny wątkowo bufor trzymający ostatnie ~N sekund próbek audio.

    Producent (callback PortAudio) dokłada bloki; gdy bufor przekroczy limit,
    najstarsze próbki są usuwane. Konsument (pipeline F12) robi migawkę.
    """

    def __init__(self, max_seconds: float, sample_rate: int) -> None:
        self._max_samples = int(max_seconds * sample_rate)
        self._blocks: "deque[np.ndarray]" = deque()
        self._count = 0
        self._lock = threading.Lock()

    def add(self, block: np.ndarray) -> None:
        with self._lock:
            self._blocks.append(block)
            self._count += len(block)
            # Usuwamy najstarsze bloki, aż zmieścimy się w limicie.
            while self._count > self._max_samples and len(self._blocks) > 1:
                self._count -= len(self._blocks.popleft())

    def snapshot(self) -> np.ndarray:
        """Zwraca kopię całego bufora jako jedną tablicę float32 (lub pustą)."""
        with self._lock:
            if not self._blocks:
                return np.empty(0, dtype=np.float32)
            return np.concatenate(list(self._blocks))


# ---------------------------------------------------------------------------
# Audio: ciągły nasłuch mikrofonu -> rolling buffer
# ---------------------------------------------------------------------------
class AudioListener:
    """Nasłuchuje mikrofonu i dopisuje bloki próbek do rolling buffera.

    Callback sounddevice działa na osobnym wątku PortAudio i jest minimalny —
    tylko kopiuje próbki do bufora.
    """

    def __init__(self, ring: AudioRingBuffer) -> None:
        self._ring = ring
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
        # Pierwszy (jedyny) kanał; KOPIUJEMY — bufor sounddevice jest reużywany.
        self._ring.add(indata[:, 0].copy())

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


# ---------------------------------------------------------------------------
# Transkrypcja w chmurze (OpenAI Whisper)
# ---------------------------------------------------------------------------
class CloudTranscriber:
    """Transkrybuje surowy dźwięk przez API OpenAI (Whisper)."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        wav = io.BytesIO(_wav_bytes_from_float32(audio, config.SAMPLE_RATE))
        wav.name = "audio.wav"  # OpenAI wykrywa format po nazwie pliku

        kwargs = {"model": self._model, "file": wav}
        if config.TRANSCRIBE_LANGUAGE:
            kwargs["language"] = config.TRANSCRIBE_LANGUAGE

        response = self._client.audio.transcriptions.create(**kwargs)
        return (response.text or "").strip()


# ---------------------------------------------------------------------------
# RAG: indeks FAISS (embeddingi OpenAI) + LLM (Claude)
# ---------------------------------------------------------------------------
class KnowledgeBase:
    """Wczytuje indeks FAISS i odpytuje Claude: wyodrębnia pytanie i generuje
    odpowiedź na podstawie znalezionych fragmentów notatek."""

    def __init__(self, anthropic_client: "anthropic.Anthropic") -> None:
        if not config.INDEX_DIR.exists():
            raise FileNotFoundError(
                f"Nie znaleziono indeksu wiedzy w {config.INDEX_DIR}.\n"
                "Uruchom najpierw:  python ingest.py"
            )
        embeddings = config.build_embeddings()
        self._vector_store = FAISS.load_local(
            str(config.INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,  # indeks budujemy lokalnie
        )
        self._retriever = self._vector_store.as_retriever(
            search_kwargs={"k": config.RETRIEVER_K}
        )
        self._client = anthropic_client

    def extract_question(self, transcript: str) -> str:
        """Wyciąga z transkrypcji najważniejsze pytanie (Claude)."""
        message = self._client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=config.ANTHROPIC_MAX_TOKENS_QUESTION,
            system=EXTRACT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": transcript}],
        )
        return _message_text(message).strip().strip('"').strip()

    def answer(self, question: str) -> "tuple[str, list[str]]":
        """Zwraca (odpowiedź, lista nazw plików źródłowych)."""
        docs = self._retriever.invoke(question)
        if not docs:
            return "Brak informacji w notatkach.", []

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        message = self._client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=config.ANTHROPIC_MAX_TOKENS_ANSWER,
            system=ANSWER_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"FRAGMENTY NOTATEK:\n{context}\n\nPYTANIE:\n{question}",
                }
            ],
        )

        # Unikalne nazwy źródeł (do pokazania pod odpowiedzią).
        sources: list[str] = []
        for doc in docs:
            raw = str(doc.metadata.get("source", "?"))
            name = raw.replace("\\", "/").split("/")[-1]
            if name not in sources:
                sources.append(name)
        return _message_text(message).strip(), sources


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
# Orkiestracja: spina audio, skrót F12 i pipeline (transkrypcja + RAG)
# ---------------------------------------------------------------------------
class Assistant:
    def __init__(self, bridge: UiBridge) -> None:
        self.bridge = bridge
        self.ring = AudioRingBuffer(config.BUFFER_SECONDS, config.SAMPLE_RATE)
        self.listener: "AudioListener | None" = None
        self.transcriber: "CloudTranscriber | None" = None
        self.kb: "KnowledgeBase | None" = None
        self._busy = threading.Lock()  # gwarantuje jeden pipeline na raz
        self._shut = False

    def initialize(self) -> None:
        """Ciężka inicjalizacja (klienci API, indeks, nasłuch, skrót).

        Uruchamiana w osobnym wątku, dzięki czemu okno GUI pojawia się od razu,
        a użytkownik widzi postęp w pasku statusu.
        """
        try:
            missing = [
                name
                for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
                if not os.environ.get(name)
            ]
            if missing:
                raise RuntimeError(
                    "Brak zmiennych środowiskowych: "
                    + ", ".join(missing)
                    + ". Ustaw je przed uruchomieniem (patrz README)."
                )

            self.bridge.status.emit("Łączenie z usługami (Claude, OpenAI)…")
            openai_client = OpenAI()
            anthropic_client = anthropic.Anthropic()
            self.transcriber = CloudTranscriber(
                openai_client, config.OPENAI_TRANSCRIBE_MODEL
            )

            self.bridge.status.emit("Wczytywanie bazy wiedzy (FAISS)…")
            self.kb = KnowledgeBase(anthropic_client)

            self.bridge.status.emit("Uruchamianie nasłuchu mikrofonu…")
            self.listener = AudioListener(self.ring)
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
        """Sekwencja F12: bufor audio -> transkrypcja -> pytanie -> RAG -> odpowiedź.

        Cała ciężka praca (sieć) dzieje się tutaj, w osobnym wątku — GUI pozostaje
        responsywne, a wyniki trafiają do okna wyłącznie przez sygnały Qt.
        """
        if not self._busy.acquire(blocking=False):
            return
        try:
            if self.transcriber is None or self.kb is None:
                self.bridge.error.emit("Asystent jeszcze się uruchamia…")
                return

            self.bridge.status.emit("Pobieram ostatnie sekundy dźwięku…")
            audio = self.ring.snapshot()
            if audio.size == 0:
                self.bridge.error.emit(
                    "Bufor pusty — mów do mikrofonu i spróbuj ponownie."
                )
                return

            self.bridge.status.emit("Transkrypcja mowy (OpenAI)…")
            transcript = self.transcriber.transcribe(audio)
            if not transcript:
                self.bridge.error.emit("Nie rozpoznano mowy. Spróbuj ponownie.")
                return

            self.bridge.status.emit("Wyodrębniam pytanie (Claude)…")
            question = self.kb.extract_question(transcript)
            if not question:
                self.bridge.error.emit("Nie udało się wyodrębnić pytania.")
                return
            self.bridge.question.emit(question)

            self.bridge.status.emit("Szukam w notatkach i generuję odpowiedź (Claude)…")
            answer, sources = self.kb.answer(question)
            self.bridge.result.emit(answer, sources)
        except anthropic.APIError as exc:
            self.bridge.error.emit(f"Błąd Claude API: {exc}")
            print(f"[pipeline] anthropic: {exc}", file=sys.stderr)
        except OpenAIError as exc:
            self.bridge.error.emit(f"Błąd OpenAI API: {exc}")
            print(f"[pipeline] openai: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            self.bridge.error.emit(f"Błąd: {exc}")
            print(f"[pipeline] {exc}", file=sys.stderr)
        finally:
            self._busy.release()

    def shutdown(self) -> None:
        """Porządne zatrzymanie nasłuchu i zwolnienie zasobów (idempotentne)."""
        if self._shut:
            return
        self._shut = True
        try:
            keyboard.unhook_all()
        except Exception:  # noqa: BLE001
            pass
        if self.listener is not None:
            self.listener.stop()


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
