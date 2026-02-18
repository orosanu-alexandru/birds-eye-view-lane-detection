from flask import Flask, Response, render_template, jsonify
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

# Parametrii pentru detectarea liniilor
line_detection_params = {
    'canny_low': 50,        # prag inferior pentru Canny
    'canny_high': 150,      # prag superior pentru Canny
    'hough_threshold': 50,  # prag pentru HoughLinesP
    'min_line_length': 50,  # lungime minimă linie
    'max_line_gap': 10,     # gap maxim între segmente
}

# Parametrii pentru gruparea liniilor
line_merge_params = {
    'angle_threshold': 15,     # prag pentru unghi (grade)
    'position_threshold': 80,  # prag pentru poziție (unități)
    'min_point_distance': 20,  # distanță minimă între puncte consecutive
}

# Stocăm liniile detectate recent
latest_detected_lines = []


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
                # Detectăm linii de marcaj
                raw_segments = detect_lane_lines(frame)
                global latest_detected_lines
                # Păstrăm segmentele brute în loc să le mergem
                latest_detected_lines = convert_segments_to_car_coords(raw_segments)
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


@app.route('/get_line_params')
def get_line_params():
    """Returnează parametrii curenți de detectare linii ca JSON."""
    return jsonify(line_detection_params)


@app.route('/set_line_params')
def set_line_params():
    """Setează parametrii de detectare linii din query params."""
    from flask import request, jsonify
    # Acceptăm parametrii ca valori int/float; dacă lipsesc, păstrăm valoarea curentă
    for key in list(line_detection_params.keys()):
        if key in request.args:
            try:
                if key in ['canny_low', 'canny_high', 'hough_threshold', 'min_line_length', 'max_line_gap']:
                    line_detection_params[key] = int(request.args.get(key))
                else:
                    line_detection_params[key] = float(request.args.get(key))
            except Exception:
                pass
    return jsonify(line_detection_params)


@app.route('/get_line_merge_params')
def get_line_merge_params():
    """Returnează parametrii curenți de grupare linii ca JSON."""
    return jsonify(line_merge_params)


@app.route('/set_line_merge_params')
def set_line_merge_params():
    """Setează parametrii de grupare linii din query params."""
    from flask import request, jsonify
    # Acceptăm parametrii ca valori int/float; dacă lipsesc, păstrăm valoarea curentă
    for key in list(line_merge_params.keys()):
        if key in request.args:
            try:
                if key in ['angle_threshold', 'position_threshold', 'min_point_distance']:
                    line_merge_params[key] = int(request.args.get(key))
                else:
                    line_merge_params[key] = float(request.args.get(key))
            except Exception:
                pass
    return jsonify(line_merge_params)


@app.route('/get_detected_lines')
def get_detected_lines():
    """Returnează liniile detectate și grupate în coordonatele mașinii."""
    return jsonify(latest_detected_lines)


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


def detect_lane_lines(frame):
    """Detectează segmente de linii de marcaj folosind Canny și Hough Transform P.

    Returnează segmentele brute ca listă de tuple (x1,y1,x2,y2) în coordonatele imaginii.
    """
    h, w = frame.shape[:2]

    # Convertim la grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicăm Gaussian blur pentru reducerea zgomotului
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, line_detection_params['canny_low'],
                     line_detection_params['canny_high'])

    # Hough Transform P pentru detectarea liniilor
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                           threshold=line_detection_params['hough_threshold'],
                           minLineLength=line_detection_params['min_line_length'],
                           maxLineGap=line_detection_params['max_line_gap'])

    raw_segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            raw_segments.append((x1, y1, x2, y2))

            # Desenăm segmentele brute cu verde deschis
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return raw_segments


def convert_segments_to_car_coords(raw_segments):
    """Convertește segmentele brute în coordonatele mașinii fără grupare."""
    if not raw_segments:
        return []

    # Folosim dimensiunile frame-ului, dar să le luăm din context
    # Pentru moment hardcodăm, dar ar trebui să fie din frame
    h, w = 480, 640

    car_lines = []
    for x1, y1, x2, y2 in raw_segments:
        car_x1 = x1 - w/2
        car_y1 = h - y1
        car_x2 = x2 - w/2
        car_y2 = h - y2
        car_lines.append([(float(car_x1), float(car_y1)), (float(car_x2), float(car_y2))])

    return car_lines


def merge_line_segments(raw_segments, frame):
    """Unește segmentele de linii care aparțin aceleiași linii continue.

    Returnează linii ca liste de puncte [x,y] în coordonatele mașinii.
    Desenează liniile merged pe frame cu roșu.
    """
    if not raw_segments:
        return []

    h, w = frame.shape[:2]

    # Convertim segmentele la coordonatele mașinii
    car_segments = []
    for x1, y1, x2, y2 in raw_segments:
        car_x1 = x1 - w/2
        car_y1 = h - y1
        car_x2 = x2 - w/2
        car_y2 = h - y2
        car_segments.append((car_x1, car_y1, car_x2, car_y2))

    # Grupăm liniile după unghi și poziție
    groups = []
    angle_threshold = line_merge_params['angle_threshold']
    position_threshold = line_merge_params['position_threshold']

    for seg in car_segments:
        x1, y1, x2, y2 = seg

        # Calculăm unghiul liniei (în grade)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle < 0:
            angle += 180  # normalizăm la 0-180

        # Poziția medie
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Căutăm un grup potrivit
        found_group = False
        for group in groups:
            group_angle = group['angle']
            group_mid_x = group['mid_x']
            group_mid_y = group['mid_y']

            # Verificăm dacă unghiurile sunt similare și pozițiile apropiate
            if (abs(angle - group_angle) < angle_threshold and
                np.sqrt((mid_x - group_mid_x)**2 + (mid_y - group_mid_y)**2) < position_threshold):
                group['segments'].append(seg)
                # Actualizăm media grupului
                total_segs = len(group['segments'])
                group['angle'] = (group_angle * (total_segs - 1) + angle) / total_segs
                group['mid_x'] = (group_mid_x * (total_segs - 1) + mid_x) / total_segs
                group['mid_y'] = (group_mid_y * (total_segs - 1) + mid_y) / total_segs
                found_group = True
                break

        if not found_group:
            groups.append({
                'angle': angle,
                'mid_x': mid_x,
                'mid_y': mid_y,
                'segments': [seg]
            })

    # Pentru fiecare grup, creăm o linie continuă
    merged_lines = []
    min_point_dist = line_merge_params['min_point_distance']
    for group in groups:
        segments = group['segments']
        if len(segments) == 1:
            # O singură segment - o păstrăm ca atare
            x1, y1, x2, y2 = segments[0]
            merged_lines.append([(x1, y1), (x2, y2)])
            # Desenăm cu roșu
            img_x1 = int(x1 + w/2)
            img_y1 = int(h - y1)
            img_x2 = int(x2 + w/2)
            img_y2 = int(h - y2)
            cv2.line(frame, (img_x1, img_y1), (img_x2, img_y2), (0, 0, 255), 3)
        else:
            # Mai multe segmente - le unim într-o linie
            # Colectăm toate punctele și le sortăm după y descrescător
            all_points = []
            for seg in segments:
                all_points.extend([(seg[0], seg[1]), (seg[2], seg[3])])

            # Sortăm după y descrescător (de la mașină spre înainte)
            all_points.sort(key=lambda p: p[1], reverse=True)

            # Eliminăm duplicatele apropiate
            filtered_points = [all_points[0]]
            for point in all_points[1:]:
                dist = np.sqrt((point[0] - filtered_points[-1][0])**2 + (point[1] - filtered_points[-1][1])**2)
                if dist > min_point_dist:  # prag pentru a evita puncte prea apropiate
                    filtered_points.append(point)

            merged_lines.append(filtered_points)

            # Desenăm linia merged cu roșu
            for i in range(len(filtered_points) - 1):
                x1, y1 = filtered_points[i]
                x2, y2 = filtered_points[i+1]
                img_x1 = int(x1 + w/2)
                img_y1 = int(h - y1)
                img_x2 = int(x2 + w/2)
                img_y2 = int(h - y2)
                cv2.line(frame, (img_x1, img_y1), (img_x2, img_y2), (0, 0, 255), 3)

    return merged_lines


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
