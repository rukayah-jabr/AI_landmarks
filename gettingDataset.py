from bing_image_downloader import downloader

landmarks = {
    "schoenbrunn_palace": "Schönbrunn Palace Vienna",
    "stephansdom": "Stephansdom Vienna",
    "riesenrad": "Wiener Riesenrad Vienna"
}

for folder_name, query in landmarks.items():
    downloader.download(query, limit=30, output_dir='data/landmarks',
                        adult_filter_off=True, force_replace=False, timeout=60)
