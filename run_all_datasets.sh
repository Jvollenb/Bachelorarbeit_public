#!/bin/bash

if [ -f .env ]; then
  echo "Lade Umgebungsvariablen aus .env-Datei..."
  export $(grep -v '^#' .env | xargs)
fi


datasets=(
    "ArticularyWordRecognition"
    "BasicMotions"
    "EMOPain"
    "Epilepsy"
    "JapaneseVowels"
    "UWaveGestureLibrary"
)

train_eps_values=(
    0.0
    0.02
)

weight_decay_values=(
    0.0
)

dropout_values=(
    0.0
)

learning_rate_values=(
    0.00003
)

use_noise_values=(
    0 #
    0.1 
)

for lr in "${learning_rate_values[@]}"; do
    for dr in "${dropout_values[@]}"; do
        for wd in "${weight_decay_values[@]}"; do
            for eps in "${train_eps_values[@]}"; do
                echo "#################################################"
                echo "Starte Durchlauf mit lr=$lr, dropout=$dr, wd=$wd, eps=$eps"
                echo "#################################################"

                for noise_val in "${use_noise_values[@]}"; do
                    echo "-------------------------------------------------"
                    echo "Aktuelle Konfiguration: LR=$lr, Dropout=$dr, WD=$wd, Train_EPS=$eps, Use_Noise=$noise_val"
                    echo "-------------------------------------------------"

                    echo "================================================="
                    echo "Starte das Training für TSCMamba."
                    echo "================================================="
                    # Schleife über alle definierten Datensätze
                    for dataset in "${datasets[@]}"; do
                        echo "-------------------------------------------------"
                        echo "Starte Experiment für Datensatz: $dataset (TSCMamba)"
                        echo "-------------------------------------------------"
                        # Rufe das TSCMamba-Skript mit allen Parametern auf
                        bash ./scripts/classification/TSCMamba.sh "$dataset" "$eps" "$noise_val" "$wd" "$dr" "$lr"
                    done

                    echo "================================================="
                    echo "Starte das Training für iTransformer."
                    echo "================================================="
                    # Schleife über alle definierten Datensätze
                    for dataset in "${datasets[@]}"; do
                        echo "-------------------------------------------------"
                        echo "Starte Experiment für Datensatz: $dataset (iTransformer)"
                        echo "-------------------------------------------------"
                        # Rufe das iTransformer-Skript mit allen Parametern auf
                        bash ./scripts/classification/iTransformer.sh "$dataset" "$eps" "$noise_val" "$wd" "$dr" "$lr"
                    done
                done
            done
        done
    done
done

# E-Mail-Notification
sender_email="jvollenbroker@gmail.com" 
receiver_email="jvollenbroker@gmail.com" 
subject="TSCMamba Experimente abgeschlossen"

echo "Sende E-Mail-Benachrichtigung..."
python3 ./send_email.py "$sender_email" "$receiver_email" "$subject"


# GitHub Upload
echo "Lade Ergebnisse zu GitHub hoch..."
git add ./csv_results/classification/* 
git commit -m "Update Ergebnisse nach Experimenten"
git push origin main 

echo "Alle Experimente abgeschlossen."
