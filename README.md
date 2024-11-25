# Laboratorium AI Forced Alignment

![Python](https://img.shields.io/badge/Python-3.10.13-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Poetry](https://img.shields.io/badge/Build-Poetry-blue.svg)

**Laboratorium AI Forced Alignment** ist ein Python-Paket zur erzwungenen Ausrichtung von Transkriptionen mit Audioaufnahmen. Es verarbeitet `.wav`-Audiodateien zusammen mit vorhandenen Transkriptionsdaten in `.json`-Dateien und generiert präzise Zeitstempel für Sätze und Wörter. Das Paket nutzt moderne Sprachmodelle und Bibliotheken wie `torchaudio` und `nltk` für effiziente und genaue Ergebnisse.

## Inhaltsverzeichnis

- [Laboratorium AI Forced Alignment](#laboratorium-ai-forced-alignment)
  - [Inhaltsverzeichnis](#inhaltsverzeichnis)
  - [Überblick](#überblick)
    - [Hauptmerkmale](#hauptmerkmale)
  - [Installation und Build](#installation-und-build)
  - [Nutzung](#nutzung)
    - [CLI-Verwendung mit Filedescriptoren](#cli-verwendung-mit-filedescriptoren)
      - [1. Modul starten und Modell laden](#1-modul-starten-und-modell-laden)
      - [2. Warten auf das "ready" Signal](#2-warten-auf-das-ready-signal)
      - [3. Dateien verarbeiten](#3-dateien-verarbeiten)
    - [Beispiel mit Shell-Skript](#beispiel-mit-shell-skript)
  - [Lizenz](#lizenz)

## Überblick

**Laboratorium AI Forced Alignment** bietet eine leistungsstarke Lösung zur Synchronisierung von Transkriptionen mit ihren entsprechenden Audioaufnahmen. Dies ist besonders nützlich für Anwendungen wie Untertitelung, Sprachforschung und automatische Inhaltsanalyse.

### Hauptmerkmale

- **Forced Alignment:** Präzise Zuordnung von Transkriptionssätzen und -wörtern zu den entsprechenden Audiozeitstempeln.
- **Modellvielfalt:** Nutzung fortschrittlicher Sprachmodelle für genaue Ergebnisse.
- **Textnormalisierung:** Umfasst die Umwandlung von Zahlen in Worte und die Anpassung von Umlauten.
- **Flexible Konfiguration:** Anpassung von Gerät (`DEVICE`) und Berechnungstyp (`COMPUTE_TYPE`) für optimale Leistung und Genauigkeit.

## Installation und Build

Dieses Paket wird mit [Poetry](https://python-poetry.org/) verwaltet. Folgen Sie diesen Schritten, um das Paket zu installieren und zu bauen:

1. **Repository klonen:**

   ```bash
   git clone https://github.com/yourusername/laboratorium-ai-forced-alignment.git
   cd laboratorium-ai-forced-alignment
   ```

2. **Abhängigkeiten installieren:**

   ```bash
   poetry install
   ```

3. **Virtuelle Umgebung aktivieren:**

   ```bash
   poetry shell
   ```

4. **Paket bauen:**

   ```bash
   poetry build
   ```

   Dieses Kommando erstellt die distributierbaren Dateien im `dist/`-Verzeichnis.

## Nutzung

Das Paket wird über die Kommandozeile (CLI) als dauerhaft laufendes Modul ausgeführt. Es ermöglicht die erzwungene Ausrichtung von Transkriptionsdaten und Audioaufnahmen mittels Filedescriptoren. Die Kommunikation erfolgt über eine Pipe, wobei das Modul "ready" sendet, sobald das Modell geladen ist und bereit zur Verarbeitung ist.

### CLI-Verwendung mit Filedescriptoren

#### 1. Modul starten und Modell laden

Starten Sie das Forced Alignment-Modul über die CLI. Das Modul lädt das Modell und signalisiert über den Filedescriptor, dass es bereit ist.

```bash
python -m laboratorium_ai_forced_alignment -f <FD>
```

**Parameter:**

- `-f`, `--fd`: Filedescriptor für die Pipe-Kommunikation.

**Beispiel:**

```bash
python -m laboratorium_ai_forced_alignment -f 3
```

#### 2. Warten auf das "ready" Signal

Nachdem das Modul gestartet wurde, lädt es das Forced Alignment-Modell. Sobald das Modell geladen ist, sendet das Modul das Signal "ready" über den angegebenen Filedescriptor.

#### 3. Dateien verarbeiten

Übergeben Sie die Pfade zur Eingabe- (`.json` für Transkription und `.wav` für Audio) und Ausgabe- (`.json` für die Ausrichtung) Datei über die Pipe. Das Modul verarbeitet die Dateien und sendet ein "done" Signal, sobald die Ausrichtung abgeschlossen ist.

**Beispiel:**

```bash
echo "path/to/input_transcript.json,path/to/input_audio.wav,path/to/output_alignment.json" >&3
```

- **Beschreibung:**
  - Das `echo`-Kommando sendet die Pfade zur Eingabe- und Ausgabedatei über den Filedescriptor `3`.
  - Das Modul empfängt die Pfade, führt die erzwungene Ausrichtung durch und speichert das Ergebnis in der `.json`-Datei.
  - Nach Abschluss sendet das Modul ein "done" Signal über den Filedescriptor.

**Vollständiger Ablauf:**

1. **Starten Sie das Forced Alignment-Modul:**

   ```bash
   python -m laboratorium_ai_forced_alignment -f 3
   ```

2. **Senden Sie die Dateiwege zur Ausrichtung:**

   ```bash
   echo "path/to/input_transcript.json,path/to/input_audio.wav,path/to/output_alignment.json" >&3
   ```

3. **Wiederholen Sie Schritt 2 für weitere Dateien:**

   ```bash
   echo "path/to/another_input_transcript.json,path/to/another_input_audio.wav,path/to/another_output_alignment.json" >&3
   ```

### Beispiel mit Shell-Skript

Hier ist ein Beispiel, wie Sie das Forced Alignment-Paket in einem Shell-Skript nutzen können:

```bash
#!/bin/bash

# Öffnen Sie einen Dateideskriptor (z.B. 3) für die Pipe-Kommunikation

exec 3<>/dev/null

# Starten Sie das Forced Alignment-Modul im Hintergrund und verbinden Sie den Filedescriptor

python -m laboratorium_ai_forced_alignment -f 3 &

# PID des Forced Alignment-Moduls speichern, um es später beenden zu können

FA_PID=$!

# Warten Sie auf das "ready" Signal

read -u 3 signal
if [ "$signal" = "ready" ]; then
echo "Modell ist bereit zur Verarbeitung."

    # Senden Sie die Eingabe- und Ausgabepfade
    echo "path/to/input_transcript.json,path/to/input_audio.wav,path/to/output_alignment.json" >&3

    # Warten Sie auf das "done" Signal
    read -u 3 signal_done
    if [ "$signal_done" = "done" ]; then
        echo "Ausrichtung abgeschlossen."
    fi

    # Weitere Ausrichtungen können hier hinzugefügt werden
    echo "path/to/another_input_transcript.json,path/to/another_input_audio.wav,path/to/another_output_alignment.json" >&3

    # Warten Sie erneut auf das "done" Signal
    read -u 3 signal_done
    if [ "$signal_done" = "done" ]; then
        echo "Weitere Ausrichtung abgeschlossen."
    fi

fi

# Schließen Sie den Filedeskriptor

exec 3>&-
```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Details finden Sie in der [LICENSE](LICENSE)-Datei.
