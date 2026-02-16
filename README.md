# Proiect Python + OpenCV - Live Camera Stream cu Flask

Scop: capturăm feed-ul camerei laptopului și îl afișăm live într-un browser.

Structura proiect:

- `server.py` - server Flask care transmite stream-ul video
- `templates/index.html` - pagină HTML care afișează feed-ul
- `requirements.txt` - dependențe Python

Instrucțiuni de rulare (Windows):

1) Creare și activare mediu virtual (opțional, recomandat):

```powershell
python -m venv venv
venv\Scripts\activate
```

2) Instalare dependențe:

```powershell
pip install -r requirements.txt
```

3) Pornire server:

```powershell
python server.py
```

4) Deschideți în browser: http://localhost:5000

Endpoint-uri utile:
- `GET /video_feed` - stream normal (fără transformare)
- `GET /video_feed_birdseye` - stream cu transformare bird's-eye (perspective)

Pe pagina principală există butoane pentru a comuta între vizualizare normală
și bird's-eye (vedere de sus).

Note:
- Dacă camera nu pornește, încercați să modificați index-ul din `cv2.VideoCapture(0)` la `1` sau `2`.
- Pentru dezvoltare, `debug=True` este activat în `server.py`; dezactivați-l în producție.

Pași următori (după streaming funcțional):
- Adăugare transformare bird's-eye view (perspective transform)
- Implementare lane detection (Canny + HoughLines sau metode moderne)
