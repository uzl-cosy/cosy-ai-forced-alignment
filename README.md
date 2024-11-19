# Laboratorium AI ASR

Dieses Repository enthält das Projekt Laboratorium AI ASR, ein Python-basiertes Projekt, das für automatische Spracherkennung (ASR) entwickelt wurde.

## Voraussetzungen

Bevor du mit der Entwicklung beginnst, stelle sicher, dass `pyenv` und `pyenv-virtualenv` auf deinem System installiert sind.

### Installation von pyenv und pyenv-virtualenv

#### Für macOS:

1. Installiere pyenv mit Homebrew:

   `brew update
brew install pyenv`

2. Füge pyenv zur `PATH`-Variable hinzu, indem du folgende Zeile in deine `.zshrc` oder `.bash_profile` einfügst:

   `export PATH="$(pyenv root)/shims:$PATH"`

3. Installiere pyenv-virtualenv:

   `brew install pyenv-virtualenv`

4. Füge die Initialisierung von pyenv-virtualenv zu deiner Shell hinzu, indem du folgende Zeile in deine `.zshrc` oder `.bash_profile` einfügst:

   `eval "$(pyenv virtualenv-init -)"`

#### Für Linux:

1. Installiere pyenv:

   `curl https://pyenv.run | bash`

2. Füge pyenv zur `PATH`-Variable hinzu, indem du folgende Zeilen in deine `.bashrc` oder `.zshrc` einfügst:

   `export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"`

3. Installiere pyenv-virtualenv, indem du es als Plugin hinzufügst:

   `git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc`

## Einrichtung der Entwicklungsumgebung

1. Klone das Repository und wechsle in das Projektverzeichnis:

   `git clone [URL-TO-REPOSITORY]
cd [REPOSITORY-NAME]`

2. Richte das virtuelle Environment mit dem bereitgestellten Skript ein:

   `./venv-setup.sh`

3. Aktiviere das virtuelle Environment:

   - Über das Terminal:

     `pyenv activate laboratorium_ai_asr_env`

   - In VSCode:

     Wähle das `laboratorium_ai_asr_env` als Python-Interpreter aus.

## Entwicklung

- Die Hauptdatei des Projekts befindet sich im Package `laboratorium_ai_asr`.
- Zugehörige Tests findest du im Verzeichnis `tests`.

## Tests ausführen

Um die Tests auszuführen, stelle sicher, dass das virtuelle Environment aktiviert ist und führe im Terminal:

`pytest`

## Abhängigkeiten speichern

Wenn neue Pakete installiert wurden, führe vor dem Commit das Skript `venv-save-dependencies.sh` aus, um die neuen Pakete aus dem virtuellen Environment in die `requirements.txt` zu extrahieren:

`./venv-save-dependencies.sh`

## Continuous Integration (CI)

Nachdem der Code ins Repository gepusht wurde, führt die CI automatisch die Tests durch. Bitte überprüfe, ob diese erfolgreich waren und bessere gegebenenfalls nach.

## Viel Erfolg bei der Entwicklung!
