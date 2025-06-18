import os
import requests
from PIL import Image
from io import BytesIO
from duckduckgo_search import DDGS
import imagehash
import time

# Einstellungen
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SAVE_DIR = os.path.join(SCRIPT_DIR, "data/landmarks_new")
SEARCH_TERMS = ["Karlskirche","Stephansdom","Schloss Belvedere","Schloss Schönbrunn","Secession","Wiener Hofburg","Wiener Riesenrad","Wiener Staatsoper","Hundertwasserhaus","DC Tower","Millenium Tower","FH Campus Wien"]  # Mehrere Suchbegriffe
NUM_IMAGES = 200
HAMMING_THRESHOLD = 0  # Toleranz für visuelle Ähnlichkeit

os.makedirs(BASE_SAVE_DIR, exist_ok=True)

with DDGS() as ddgs:
    for term in SEARCH_TERMS:
        max_retries = 3
        attempt = 0
        success = False
        while attempt < max_retries and not success:
            try:
                # Ordner pro Suchbegriff anlegen (Leerzeichen durch Unterstrich ersetzen)
                safe_term = term.replace(" ", "_")
                SAVE_DIR = os.path.join(BASE_SAVE_DIR, safe_term)
                os.makedirs(SAVE_DIR, exist_ok=True)
                known_hashes = []

                # Vorhandene Bilder scannen und Hashes laden
                for file in os.listdir(SAVE_DIR):
                    path = os.path.join(SAVE_DIR, file)
                    try:
                        with Image.open(path) as img:
                            hash_val = imagehash.phash(img.convert("RGB"))
                            known_hashes.append(hash_val)
                    except:
                        continue

                search_term = term + " wien vienna bild außen image outside"
                # Neue Bilder suchen und speichern
                results = ddgs.images(search_term, max_results=NUM_IMAGES)
                results = list(results)

                for result in results:
                    url = result["image"]
                    try:
                        response = requests.get(url, timeout=15)
                        response.raise_for_status()
                        img = Image.open(BytesIO(response.content)).convert("RGB")
                        img_hash = imagehash.phash(img)

                        # Visuelle Duplikaterkennung
                        if any(existing_hash - img_hash <= HAMMING_THRESHOLD for existing_hash in known_hashes):
                            print(f"[{term}] Ähnliches Bild erkannt – übersprungen: {url}")
                            continue

                        filename = os.path.join(SAVE_DIR, f"{img_hash}.jpg")
                        img.save(filename)
                        known_hashes.append(img_hash)
                        print(f"[{term}] Bild gespeichert: {filename}")

                    except Exception as e:
                        print(f"[{term}] Fehler bei {url}: {e}")
                    time.sleep(1)  # 1 Sekunde Pause zwischen den Downloads

                # Nach jedem Suchbegriff eine Pause einlegen
                print(f"[{term}] Warte 30 Sekunden, um Rate Limits zu vermeiden...")
                time.sleep(30)
                success = True  # Wenn alles geklappt hat, Schleife verlassen

            except Exception as e:
                attempt += 1
                print(f"[{term}] Fehler beim gesamten Durchlauf (Versuch {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    print(f"[{term}] Warte 60 Sekunden vor erneutem Versuch...")
                    time.sleep(60*attempt)  # Wartezeit erhöht sich mit jedem Versuch
                else:
                    print(f"[{term}] Max. Versuche erreicht, überspringe diesen Suchbegriff.")
