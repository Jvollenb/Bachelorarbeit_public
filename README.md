# Untersuchung der Robustheit von TSCMamba für die Zeitreihenklassifikation gegenüber adversarialen Angriffen

Dieses Repository enthält den vollständigen Quellcode und die Konfigurationsdateien für die Bachelorarbeit mit dem Titel „Untersuchung der Robustheit von TSCMamba für die Zeitreihenklassifikation gegenüber adversarialen Angriffen“, eingereicht an der Universität Münster.

Die Arbeit evaluiert systematisch die Robustheit des Mamba-basierten Modells TSCMamba im Vergleich zur etablierten iTransformer-Architektur. Es wird die Anfälligkeit gegenüber White-Box-Angriffen (FGSM, PGD) quantifiziert und die Wirksamkeit von Verteidigungsstrategien (Adversarial Training, zufälliges Rauschen) untersucht.

## Struktur des Repositories

-   `run_all_datasets.sh`: Bash-Skript zur automatisierten Durchführung aller Experimente über alle Datensätze hinweg.
-   `run.py`: Hauptskript zur Ausführung der Trainings- und Evaluationsprozesse.
-   `models/`: Enthält die Python-Implementierungen der Modelle `TSCMamba` und `iTransformer`.
-   `adversarials/`: Implementierung der adversarialen Angriffe FGSM und PGD.
-   `data_provider/`: Skripte für das Laden und Vorverarbeiten der Datensätze.
-   `configs/`: Konfigurationsdateien im JSON-Format zur Steuerung der Experimente.
-   `results/`: Verzeichnis, in dem die generierten Ergebnisdateien (Logs, Plots) gespeichert werden.
-   `requirements.txt`: Liste aller notwendigen Python-Pakete und deren Versionen.

## Installation und Vorbereitung der Umgebung

Für eine exakte Reproduktion der Ergebnisse ist die folgende Vorgehensweise zwingend erforderlich.

1.  **Klonen Sie das Repository:**
    ```bash
    git clone https://github.com/Jvollenb/Bachelorarbeit.git
    cd Bachelorarbeit
    ```

2.  **Erstellen einer virtuellen Umgebung:** Es wird dringend empfohlen, eine dedizierte Python-Umgebung zu verwenden, um Konflikte zwischen Paketversionen zu vermeiden.
    ```bash
    python -m venv venv
    ```

3.  **Aktivieren der Umgebung:**
    *   Für macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   Für Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Installieren der Abhängigkeiten:** Alle erforderlichen Pakete sind in der Datei `requirements.txt` spezifiziert.
    ```bash
    pip install -r requirements.txt
    ```
    *Hinweis: Die Experimente wurden unter Python 3.12.3 und CUDA 12.1 durchgeführt. Die Verwendung abweichender Versionen kann zu unterschiedlichen Ergebnissen führen.*

## Durchführung der Experimente

Die Experimente werden über das Hauptskript `run.py` gesteuert. Die spezifische Konfiguration für ein Experiment (Modellwahl, Datensatz, Angriffsparameter, Verteidigungsstrategie) wird über das Bash-Skript `run_all_datasets.sh` oder direkt über die Kommandozeile festgelegt.

Um die in der Arbeit beschriebenen Experimente zu reproduzieren, führen Sie das Bash-Skript:
```bash
./run_all_datasets.sh
```

## Ergebnisse

Nach erfolgreicher Ausführung werden die Log-Dateien und Grafiken im `results/`-Verzeichnis abgelegt. Die in der Arbeit präsentierten und diskutierten Kernergebnisse finden sich in Kapitel 4 sowie im Anhang (Tabelle 7.1, 7.2) der PDF-Version der Bachelorarbeit.

## Zitation

Sollten Sie diese Arbeit oder den dazugehörigen Code in Ihrer eigenen Forschung verwenden, wird die Verwendung des folgenden BibTeX-Eintrags empfohlen:

```bibtex
@bachelorthesis{Vollenbroker2025,
  author  = {Julius Maximilian Vollenbröker},
  title   = {Untersuchung der Robustheit von {TSCMamba} für die Zeitreihenklassifikation gegenüber adversarialen Angriffen},
  school  = {Universität Münster},
  year    = {2025},
  month   = {7},
  note    = {Verfügbar unter: https://github.com/Jvollenb/Bachelorarbeit_public}
}
```

## Lizenz

Dieses Projekt steht unter der MIT Lizenz.

## Kontakt

Für wissenschaftliche Anfragen kontaktieren Sie bitte Julius Maximilian Vollenbröker (jvollenb@uni-muenster.de).

