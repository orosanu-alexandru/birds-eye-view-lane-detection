from flask import Flask, Response, render_template
import cv2
import time
import atexit
import numpy as np

# Aplicatie Flask
app = Flask(__name__)

# Deschidem camera laptopului (index 0)
# Pentru Windows, OpenCV folosește de obicei backendul implicit; dacă nu pornește,
# încercați alte index-uri (1,2) sau setări de backend.
video_capture = cv2.VideoCapture(0)

# Parametrii transformării (valori relative) - pot fi ajustați din UI
transform_params = {
    'top_width': 0.53,      # lățimea dreptunghiului de sus ca fracțiune din w (centered)
    'bottom_width': 0.90,   # lățimea la bază a sursei (fracțiune din w)
    'top_y': 0.55,          # poziția y (fracțiune din h) pentru linia de sus a trapezei
    'bottom_y': 0.95,       # poziția y pentru linia de jos a trapezei
    'center_x': 0.5,        # centru orizontal al trapezei (fracțiune din w)
    'rotate_deg': 0.0,      # rotație finală (grade) - 0,90,180,270
}


def generate_frames(mode='normal'):
    """Generator care captează frame-uri de la camera și le encodează ca JPEG.

    Flask poate trimite acest stream către browser cu mime-type-ul
    'multipart/x-mixed-replace', astfel browserul va afișa un flux video live.
    """
    while True:
        # Citim un frame de la camera
        success, frame = video_capture.read()
        if not success:
            # Dacă nu se poate citi frame-ul, așteptăm puțin și încercăm din nou
            time.sleep(0.1)
            continue

        # Opcțional: redimensionare sau pre-procesare aici pentru performanță
        # frame = cv2.resize(frame, (640, 480))

        # Dacă modul este 'birdseye', aplicăm transformarea perspectivei
        if mode == 'birdseye':
            try:
                frame = apply_perspective_transform(frame)
            except Exception:
                # Dacă transformarea eșuează, continuăm cu frame-ul original
                pass

        # Encode-ăm frame-ul ca JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # Construim părțile multipart pentru stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Pagina principală care încarcă template-ul HTML."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Endpoint care returnează stream-ul video.

    Browserul va cere această adresă ca sursă pentru un tag <img> și va primi
    un flux multipart JPEG actualizat continuu.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_birdseye')
def video_feed_birdseye():
    """Endpoint pentru stream cu transformare bird's-eye aplicată."""
    return Response(generate_frames(mode='birdseye'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_transform')
def get_transform():
    """Returnează parametrii curenți de transformare ca JSON."""
    from flask import jsonify
    return jsonify(transform_params)


@app.route('/set_transform')
def set_transform():
    """Setează parametrii de transformare din query params.

    Exemplu: /set_transform?top_width=0.12&bottom_width=0.9&top_y=0.6
    """
    from flask import request, jsonify
    # Acceptăm parametrii ca valori float; dacă lipsesc, păstrăm valoarea curentă
    for key in list(transform_params.keys()):
        if key in request.args:
            try:
                transform_params[key] = float(request.args.get(key))
            except Exception:
                pass
    return jsonify(transform_params)


def apply_perspective_transform(frame):
    """Aplică o transformare perspective (bird's-eye) asupra imaginii.

    Observații pentru începători:
    - Alegem 4 puncte sursă (`src`) care formează o trapeză în partea de jos
      a imaginii (unde de obicei se află banda de drum).
    - Le mapăm pe 4 puncte destinație (`dst`) care formează un dreptunghi.
    - `cv2.getPerspectiveTransform` calculează matricea de transformare, iar
      `cv2.warpPerspective` aplică transformarea.

    Punctele sunt estimate relativ la dimensiunea imaginii; poate fi necesară
    ajustarea lor pentru camera specifică sau pentru scenariul tău.
    """
    h, w = frame.shape[:2]

    # Calculăm punctele sursă pornind de la parametrii relativi
    cx = transform_params['center_x'] * w
    top_w = transform_params['top_width'] * w
    bot_w = transform_params['bottom_width'] * w
    top_y = transform_params['top_y'] * h
    bot_y = transform_params['bottom_y'] * h

    # colțuri sursă: stânga sus, dreapta sus, dreapta jos, stânga jos
    src = np.float32([
        [cx - top_w / 2.0, top_y],
        [cx + top_w / 2.0, top_y],
        [cx + bot_w / 2.0, bot_y],
        [cx - bot_w / 2.0, bot_y],
    ])

    # Puncte destinație - un dreptunghi complet (vedere de sus)
    dst = np.float32([
        [w * 0.2, 0],
        [w * 0.8, 0],
        [w * 0.8, h],
        [w * 0.2, h],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)

    # Aplicăm rotație dacă este specificată (valori comune: 0,90,180,270)
    deg = transform_params.get('rotate_deg', 0.0) % 360
    if deg != 0:
        # center, scale 1.0
        center = (w // 2, h // 2)
        R = cv2.getRotationMatrix2D(center, deg, 1.0)
        warped = cv2.warpAffine(warped, R, (w, h), flags=cv2.INTER_LINEAR)

    return warped


def release_resources():
    """Eliberăm resursele la terminare (camera)."""
    try:
        if video_capture.isOpened():
            video_capture.release()
    except Exception:
        pass


if __name__ == '__main__':
    # Înregistrăm funcția de cleanup
    atexit.register(release_resources)

    # Pornim serverul Flask. Accesați: http://localhost:5000
    # threaded=True permite gestionarea cererilor în fire separate.
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
