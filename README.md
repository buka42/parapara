# 🧠 Półautomatyczny asystent wiedzy (wersja chmurowa)

Lekka aplikacja desktopowa, która **nasłuchuje mowę w tle**, trzyma w pamięci
surowy dźwięk z ostatnich ~30 sekund (*rolling buffer*), a po wciśnięciu **F12**
transkrybuje go, wyciąga pytanie i generuje zwięzłą odpowiedź **wyłącznie na
podstawie Twoich notatek** (RAG).

Ta wersja jest oparta o chmurę, więc **działa płynnie nawet na słabym
komputerze** — lokalnie dzieje się tylko przechwytywanie dźwięku z mikrofonu,
a całe „ciężkie” przetwarzanie jest zdalne:

| Element | Gdzie | Technologia |
|---|---|---|
| LLM (pytanie + odpowiedź) | ☁️ chmura | **Claude Haiku 4.5** (Anthropic API) |
| Transkrypcja mowy | ☁️ chmura | **OpenAI Whisper** (`whisper-1`) |
| Embeddingi (RAG) | ☁️ chmura | **OpenAI** (`text-embedding-3-small`) |
| Baza wektorowa | 💻 lokalnie | FAISS |
| Przechwytywanie audio + GUI | 💻 lokalnie | sounddevice + PyQt6 |

> **Uwaga dot. odpowiedzialnego użycia.** Narzędzie powstało jako prywatny
> asystent do **nauki, powtórek i pracy z własnymi materiałami**. Używaj go
> wyłącznie tam, gdzie jest to dozwolone, i z poszanowaniem zasad (np.
> egzaminów) oraz prywatności osób, których głos mógłby być nagrywany.
>
> **Prywatność (wersja chmurowa).** Dźwięk z ostatnich ~30 s jest wysyłany do
> OpenAI (transkrypcja), a wyodrębnione pytanie oraz pasujące fragmenty notatek
> trafiają do OpenAI (embeddingi) i Anthropic (odpowiedź). Jeśli to problem,
> wróć do wersji w pełni lokalnej (Ollama + faster-whisper).

---

## 1. Wymagania wstępne

- **Python 3.10+**
- **Klucz Anthropic** → https://console.anthropic.com (zmienna `ANTHROPIC_API_KEY`)
- **Klucz OpenAI** → https://platform.openai.com (zmienna `OPENAI_API_KEY`)
- **PortAudio** – biblioteka systemowa dla `sounddevice` (przechwytywanie mikrofonu):
  - Linux (Debian/Ubuntu): `sudo apt install portaudio19-dev`
  - macOS: `brew install portaudio`
  - Windows: instaluje się wraz z `sounddevice` (brak dodatkowych kroków)

---

## 2. Instalacja

```bash
# 1) (zalecane) środowisko wirtualne
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) zależności Pythona
pip install -r requirements.txt

# 3) klucze API (w tej samej sesji terminala, w której uruchamiasz program)
export ANTHROPIC_API_KEY=sk-ant-...     # Windows: setx ANTHROPIC_API_KEY sk-ant-...
export OPENAI_API_KEY=sk-...            # Windows: setx OPENAI_API_KEY sk-...
```

---

## 3. Użycie

### Krok 1 — przygotowanie bazy wiedzy (Moduł A)
Wrzuć pliki `.txt`, `.md` lub `.pdf` do folderu **`notatki/`** (jest tam już
`przyklad.txt`), a następnie zbuduj indeks:

```bash
python ingest.py
```

Skrypt potnie tekst na fragmenty, policzy embeddingi (OpenAI) i zapisze indeks
wektorowy w `faiss_index/`. Powtarzaj ten krok po każdej zmianie notatek.

### Krok 2 — uruchomienie asystenta (Moduł B)
```bash
python main.py
```
- Pojawi się małe, **pływające okno** (zawsze na wierzchu).
- Aplikacja zacznie nasłuchiwać mikrofonu (bufor ostatnich ~30 s dźwięku).
- W dowolnym momencie wciśnij **F12** — asystent prześle ostatnie sekundy do
  transkrypcji, wyodrębni pytanie, przeszuka notatki i pokaże odpowiedź.
- Okno można **przeciągać** myszą, **Esc** chowa je, a **✕** kończy działanie.

> **Linux:** biblioteka `keyboard` wymaga uprawnień administratora do
> przechwytywania globalnych klawiszy — uruchom `sudo -E python main.py`
> (`-E` zachowuje zmienne środowiskowe z kluczami API). Pod **Wayland**
> globalne skróty mogą nie działać — rozważ sesję X11.
>
> **macOS:** nadaj terminalowi/Pythonowi uprawnienia *Accessibility* oraz dostęp
> do mikrofonu (Ustawienia → Prywatność i bezpieczeństwo).

---

## 4. Koszty

Jeden cykl F12 to mniej więcej:
- 1 transkrypcja ~30 s audio (OpenAI Whisper, ~$0.006/min → grosze),
- 2 krótkie wywołania Claude Haiku 4.5 (najtańszy model),
- 1 drobne wyliczenie embeddingu pytania.

W praktyce to ułamki centa za pytanie. Embeddingi notatek liczone są raz przy
`ingest.py`. Realne kwoty sprawdzaj w panelach Anthropic i OpenAI.

---

## 5. Konfiguracja i dostrajanie

Parametry są w **`config.py`**:

| Parametr | Znaczenie | Wskazówka |
|---|---|---|
| `BUFFER_SECONDS` | długość bufora dźwięku (s) | 15–30 s w zupełności wystarcza |
| `ANTHROPIC_MODEL` | model Claude | `claude-haiku-4-5` (szybko/tanio); `claude-sonnet-4-6` lub `claude-opus-4-8` dla wyższej jakości |
| `OPENAI_TRANSCRIBE_MODEL` | model transkrypcji | `whisper-1` |
| `OPENAI_EMBED_MODEL` | model embeddingów | `text-embedding-3-small` (tanio) / `-large` (dokładniej) |
| `TRANSCRIBE_LANGUAGE` | język mowy | `"pl"`; `None` = auto-detekcja |
| `RETRIEVER_K` | liczba fragmentów do kontekstu | 3–6 |
| `HOTKEY` | klawisz wyzwalacza | np. `"f12"` |

Zmiana `OPENAI_EMBED_MODEL` wymaga ponownego `python ingest.py` (indeks musi być
policzony tym samym modelem, którym potem odpytujemy).

---

## 6. Jak rozwiązano wielowątkowość (bufor audio vs. GUI)

Reguła: **nagrywanie nigdy nie dzieje się na wątku GUI, a GUI nigdy nie jest
aktualizowane z wątków roboczych.**

```
   mikrofon
      │  (wątek PortAudio – callback, krótki i nieblokujący)
      ▼
 AudioRingBuffer  ── bezpieczny wątkowo bufor ostatnich ~30 s (deque + Lock)
      ▲
      │  snapshot()
      │
 F12 → wątek keyboard → wątek "Pipeline"
                              │  (wolne operacje sieciowe — NIE na wątku GUI)
                              │   OpenAI Whisper → Claude → FAISS → Claude
                              ▼  sygnały Qt (połączenie kolejkowane)
                       wątek główny / GUI (PyQt6)  ── tylko tu rysujemy okno
```

Dlaczego to **eliminuje typowe problemy** (zacinanie okna, wyścigi, „nakładanie
się” nagrywania na obsługę GUI):

- **Callback audio jest minimalny** — tylko kopiuje próbki do bufora chronionego
  `Lock`iem. Nasłuch nigdy nie blokuje GUI.
- **Bufor surowego dźwięku zamiast ciągłej transkrypcji.** Transkrypcja jest w
  chmurze, więc transkrybujemy *na żądanie* (po F12), a nie bez przerwy — mniej
  ruchu, niższy koszt, mniejsza złożoność i brak osobnego wątku transkrypcji.
- **Ciężki pipeline (F12) działa w osobnym wątku**, a wynik wraca do okna
  **wyłącznie przez sygnały Qt** — Qt sam marshaluje je do wątku głównego, więc
  widżety są dotykane tylko z wątku GUI (wymóg PyQt).
- **Blokada `_busy`** sprawia, że szybkie, wielokrotne wciśnięcia F12 nie
  uruchamiają kilku analiz naraz.

### Dostrajanie bufora
- Gdy widzisz w konsoli ostrzeżenia `[audio] status: ...` (przepełnienia),
  zwykle nic złego się nie dzieje — to chwilowe zadławienia systemu audio.
- Krótszy `BUFFER_SECONDS` = mniej dźwięku do wysłania (szybsza, tańsza
  transkrypcja), ale mniej kontekstu; dłuższy = odwrotnie.

---

## 7. Pliki w projekcie

| Plik | Rola |
|---|---|
| `config.py` | wspólna konfiguracja + fabryka embeddingów |
| `ingest.py` | **Moduł A** — budowa indeksu FAISS z notatek |
| `main.py` | **Moduł B** — nasłuch, F12, transkrypcja, RAG, okno |
| `requirements.txt` | zależności Pythona |
| `notatki/` | Twoje materiały (`.txt`, `.md`, `.pdf`) |
| `faiss_index/` | wygenerowany indeks wektorowy (nie wersjonowany) |

---

## 8. Najczęstsze problemy

| Objaw | Przyczyna / rozwiązanie |
|---|---|
| `Brak zmiennych środowiskowych: ...` | ustaw `ANTHROPIC_API_KEY` i `OPENAI_API_KEY` w tej samej sesji |
| `Nie znaleziono indeksu wiedzy` | uruchom najpierw `python ingest.py` |
| `Błąd Claude API` / `Błąd OpenAI API` | sprawdź poprawność kluczy, limity konta i połączenie z siecią |
| Brak reakcji na F12 (Linux) | uruchom `sudo -E python main.py`; pod Wayland przełącz się na X11 |
| `PortAudioError` / brak mikrofonu | zainstaluj PortAudio i sprawdź domyślne urządzenie wejściowe |
| Pusty bufor / „Nie rozpoznano mowy” | mów wyraźnie do mikrofonu; daj kilka sekund nagrania przed F12 |
