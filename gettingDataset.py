from bing_image_downloader import downloader

landmarks = {
    "schoenbrunn_palace": "Schönbrunn Palace Vienna outside architecture",
    "stephansdom": "Stephansdom Vienna outside architecture",
    "riesenrad": "Wiener Riesenrad Vienna ferris wheel",
    "karlskirche": "Karlskirche Vienna outside karlsplatz"
}

for folder_name, query in landmarks.items():
    downloader.download(query, limit=60, output_dir=f'data/landmarks60',
                        adult_filter_off=True, force_replace=False, timeout=60)
