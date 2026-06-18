# 🧠 Półautomatyczny asystent wiedzy

Lokalna (offline) aplikacja desktopowa, która **nasłuchuje mowę w tle**,
trzyma w pamięci tekst z ostatnich ~30 sekund (*rolling buffer*), a po
wciśnięciu **F12** wyciąga z niego pytanie i generuje zwięzłą odpowiedź
**wyłącznie na podstawie Twoich notatek** (RAG).

Wszystko działa lokalnie: transkrypcja (`faster-whisper`), baza wiedzy
(`FAISS`) i model językowy (`Ollama`). Nic nie wychodzi do chmury.

> **Uwaga dot. odpowiedzialnego użycia.** Narzędzie powstało jako prywatny
> asystent do **nauki, powtórek i pracy z własnymi materiałami** (np. szybkie
> odnajdywanie informacji podczas prezentacji, spotkań czy przygotowań).
> Używaj go wyłącznie tam, gdzie jest to dozwolone, i z poszanowaniem zasad
> (np. egzaminów) oraz prywatności osób, których głos mógłby być nagrywany.

---

## 1. Stos technologiczny

| Element | Technologia |
|---|---|
| Transkrypcja offline | `faster-whisper` |
| Baza wiedzy (RAG) | `langchain` + `FAISS` |
| Embeddingi + LLM | `Ollama` (np. `llama3` + `nomic-embed-text`) |
| Globalny skrót | `keyboard` |
| GUI (pływające okno) | `PyQt6` |
| Audio | `sounddevice` (PortAudio) |

---

## 2. Wymagania wstępne

- **Python 3.10+**
- **Ollama** – https://ollama.com (serwer modeli LLM, działa lokalnie)
- **PortAudio** – biblioteka systemowa dla `sounddevice`:
  - Linux (Debian/Ubuntu): `sudo apt install portaudio19-dev`
  - macOS: `brew install portaudio`
  - Windows: instaluje się wraz z `sounddevice` (brak dodatkowych kroków)

---

## 3. Instalacja

```bash
# 1) (zalecane) środowisko wirtualne
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) zależności Pythona
pip install -r requirements.txt

# 3) Ollama – pobierz modele (jednorazowo)
ollama pull llama3                 # model do pytań i odpowiedzi
ollama pull nomic-embed-text       # model do embeddingów (RAG)

# 4) upewnij się, że serwer Ollama działa
ollama serve                       # (zwykle startuje automatycznie po instalacji)
```

---

## 4. Użycie

### Krok 1 — przygotowanie bazy wiedzy (Moduł A)
Wrzuć swoje pliki `.txt`, `.md` lub `.pdf` do folderu **`notatki/`**
(jest tam już `przyklad.txt`), a następnie zbuduj indeks:

```bash
python ingest.py
```

Skrypt potnie tekst na fragmenty i zapisze indeks wektorowy w `faiss_index/`.
Powtarzaj ten krok po każdej zmianie notatek.

### Krok 2 — uruchomienie asystenta (Moduł B)
```bash
python main.py
```
- Pojawi się małe, **pływające okno** (zawsze na wierzchu).
- Aplikacja zacznie nasłuchiwać mikrofonu i transkrybować mowę w tle.
- W dowolnym momencie wciśnij **F12** — asystent przeanalizuje ostatnie
  sekundy, wyodrębni pytanie, przeszuka notatki i pokaże odpowiedź.
- Okno można **przeciągać** (chwyć je myszą), **Esc** chowa je, a **✕** kończy
  działanie aplikacji.

> **Linux:** biblioteka `keyboard` wymaga uprawnień administratora do
> przechwytywania globalnych klawiszy — uruchom `sudo python main.py`
> (w środowisku wirtualnym: `sudo .venv/bin/python main.py`). Pod **Wayland**
> globalne skróty mogą nie działać — rozważ sesję X11.
>
> **macOS:** nadaj terminalowi/Pythonowi uprawnienia *Accessibility*
> (Ustawienia → Prywatność i bezpieczeństwo → Dostępność) oraz dostęp do
> mikrofonu. Globalny skrót może wymagać uruchomienia z `sudo`.

---

## 5. Konfiguracja i dostrajanie

Wszystkie parametry są w **`config.py`**:

| Parametr | Znaczenie | Wskazówka |
|---|---|---|
| `BUFFER_SECONDS` | długość rolling buffera (s) | 15–30 s w zupełności wystarcza |
| `CHUNK_SECONDS` | co ile sekund transkrybujemy paczkę | mniej = świeższy bufor, ale gorsza jakość; więcej = dokładniej, ale „starszy” tekst |
| `WHISPER_MODEL` | rozmiar modelu Whisper | `tiny`/`base` = szybko, `small`/`medium` = dokładniej |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE` | CPU/GPU | GPU: `"cuda"` + `"float16"` |
| `OLLAMA_MODEL` | model LLM | dowolny zainstalowany w Ollamie |
| `RETRIEVER_K` | liczba fragmentów do kontekstu | 3–6 |
| `HOTKEY` | klawisz wyzwalacza | np. `"f12"` |

Pierwsze uruchomienie `main.py` pobierze model Whisper (potrzebny jednorazowy
dostęp do Internetu); potem działa w pełni offline.

---

## 6. Jak rozwiązano wielowątkowość (bufor audio vs. GUI)

To była kluczowa część zadania. Reguła brzmi: **nagrywanie nigdy nie dzieje
się na wątku GUI, a GUI nigdy nie jest aktualizowane z wątków roboczych.**
Architektura ma cztery rozdzielone warstwy:

```
   mikrofon
      │  (wątek PortAudio – callback, krótki i nieblokujący)
      ▼
 queue.Queue  ── thread-safe bufor próbek (producent → konsument)
      │
      ▼
 RollingTranscriber (osobny wątek)
      │  faster-whisper transkrybuje paczki po CHUNK_SECONDS
      ▼
 deque[(czas, tekst)]  ── rolling buffer chroniony Lockiem; stare wpisy usuwane
      ▲
      │  get_transcript()
      │
 F12 → wątek keyboard → wątek "Pipeline"  (LLM + RAG; wolne operacje)
                              │
                              ▼  sygnały Qt (połączenie kolejkowane)
                       wątek główny / GUI (PyQt6)  ── tylko tu rysujemy okno
```

Dlaczego to **eliminuje typowe problemy** (zacinanie się okna, wyścigi,
„nakładanie się” nagrywania na obsługę GUI):

- **Callback audio jest minimalny** — tylko kopiuje próbki do kolejki.
  Cała transkrypcja idzie do osobnego wątku, więc nasłuch nigdy nie blokuje
  ani GUI, ani siebie samego.
- **`queue.Queue` + `deque` z `Lock`** synchronizują dostęp do współdzielonych
  danych. Gdy transkrypcja nie nadąża, callback **porzuca najstarsze** próbki,
  by trzymać się „na żywo” zamiast budować rosnące opóźnienie.
- **Ciężki pipeline (F12) działa w osobnym wątku**, a wynik wraca do okna
  **wyłącznie przez sygnały Qt** — Qt sam marshaluje je do wątku głównego, więc
  widżety są dotykane tylko z wątku GUI (wymóg PyQt/większości toolkitów).
- **Blokada `_busy`** sprawia, że szybkie, wielokrotne wciśnięcia F12 nie
  uruchamiają kilku analiz naraz.

### Dostrajanie bufora, by uniknąć kłopotów
- Jeśli słyszysz „przepełnienia” (`[audio] status: ...`) lub bufor jest
  przestarzały na słabszym sprzęcie — wybierz mniejszy `WHISPER_MODEL`
  (np. `base`), zwiększ `CHUNK_SECONDS` albo użyj GPU.
- `AUDIO_QUEUE_MAXSIZE` ogranicza zużycie pamięci, gdy transkrypcja zwalnia.
- Świeżość bufora zależy od `CHUNK_SECONDS`: w skrajnym przypadku najnowsza,
  jeszcze nieprzetworzona paczka (do `CHUNK_SECONDS` s mowy) nie znajdzie się
  jeszcze w buforze — dlatego F12 warto wcisnąć chwilę PO usłyszanym pytaniu.

---

## 7. Pliki w projekcie

| Plik | Rola |
|---|---|
| `config.py` | wspólna konfiguracja (modele, ścieżki, parametry bufora) |
| `ingest.py` | **Moduł A** — budowa bazy wiedzy z notatek |
| `main.py` | **Moduł B** — nasłuch, transkrypcja, F12, RAG, okno |
| `requirements.txt` | zależności Pythona |
| `notatki/` | Twoje materiały (`.txt`, `.md`, `.pdf`) |
| `faiss_index/` | wygenerowany indeks wektorowy (nie wersjonowany) |

---

## 8. Najczęstsze problemy

| Objaw | Przyczyna / rozwiązanie |
|---|---|
| `Nie znaleziono indeksu wiedzy` | uruchom najpierw `python ingest.py` |
| Błąd embeddingów / połączenia | uruchom `ollama serve` i `ollama pull nomic-embed-text` |
| Brak reakcji na F12 (Linux) | uruchom z `sudo`; pod Wayland przełącz się na X11 |
| `PortAudioError` / brak mikrofonu | zainstaluj PortAudio i sprawdź domyślne urządzenie wejściowe |
| Wolne odpowiedzi | mniejszy `WHISPER_MODEL`/`OLLAMA_MODEL` lub użyj GPU |
| Pusty bufor po F12 | mów wyraźnie do mikrofonu; daj kilka sekund na transkrypcję |
