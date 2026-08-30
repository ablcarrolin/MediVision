import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Inisialisasi MediaPipe Holistic Utuh
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def hitung_sudut(a, b, c):
    """Menhitung sudut 3D/2D antar sendi dalam derajat"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return round(angle, 1)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    # UBAH INDEKS KAMERA DI SINI JIKA PAKAI WEBCAM USB (misal: 1 atau 2)
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        # 1. VISUALISASI UTUH PADA VIDEO STREAM
        
        # A. Deteksi Wajah Detail (468 Titik Face Mesh)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        # B. Deteksi Tubuh (Pose 33 Titik)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2)
            )

        # C. Deteksi Jari Tangan Kiri (21 Titik)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # D. Deteksi Jari Tangan Kanan (21 Titik)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # 2. KALKULASI DATA KESEHATAN & SUDUT (Kirim via SocketIO)
        data_web = {
            "siku_kiri": 0,
            "pinggang_kiri": 0,
            "kemiringan_bahu": 0,
            "status_postur": "Mencari Objek...",
            "skor_kesehatan": 0,
            "wajah_terdeteksi": False,
            "tangan_kiri_terdeteksi": False,
            "tangan_kanan_terdeteksi": False
        }

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Koordinat Utama
            bahu_kiri = [lm[mp_holistic.PoseLandmark.LEFT_SHOULDER].x, lm[mp_holistic.PoseLandmark.LEFT_SHOULDER].y]
            bahu_kanan = [lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y]
            siku_kiri = [lm[mp_holistic.PoseLandmark.LEFT_ELBOW].x, lm[mp_holistic.PoseLandmark.LEFT_ELBOW].y]
            pergelangan_kiri = [lm[mp_holistic.PoseLandmark.LEFT_WRIST].x, lm[mp_holistic.PoseLandmark.LEFT_WRIST].y]
            pinggul_kiri = [lm[mp_holistic.PoseLandmark.LEFT_HIP].x, lm[mp_holistic.PoseLandmark.LEFT_HIP].y]
            lutut_kiri = [lm[mp_holistic.PoseLandmark.LEFT_KNEE].x, lm[mp_holistic.PoseLandmark.LEFT_KNEE].y]

            # Hitung Sudut-Sudut
            sudut_siku = hitung_sudut(bahu_kiri, siku_kiri, pergelangan_kiri)
            sudut_pinggang = hitung_sudut(bahu_kiri, pinggul_kiri, lutut_kiri)
            
            # Hitung Kemiringan Bahu (Simetri Tulang Belakang)
            selisih_bahu = round(abs(bahu_kiri[1] - bahu_kanan[1]) * 100, 1)

            # Algoritma Penilaian Kesehatan Postur Visual
            skor = 100
            status = "Sangat Baik (Tegak)"

            if selisih_bahu > 4.0:
                skor -= 20
                status = "Bahu Asimetris (Miring)"
            
            if sudut_pinggang < 150:  # Posisi membungkuk
                skor -= 30
                status = "Postur Bungkuk!"

            data_web["siku_kiri"] = sudut_siku
            data_web["pinggang_kiri"] = sudut_pinggang
            data_web["kemiringan_bahu"] = selisih_bahu
            data_web["skor_kesehatan"] = max(0, skor)
            data_web["status_postur"] = status

        data_web["wajah_terdeteksi"] = True if results.face_landmarks else False
        data_web["tangan_kiri_terdeteksi"] = True if results.left_hand_landmarks else False
        data_web["tangan_kanan_terdeteksi"] = True if results.right_hand_landmarks else False

        # Kirim data JSON ke Web
        socketio.emit('update_data', data_web)

        # Stream Frame Video Utuh
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Server Berjalan! Buka browser: http://127.0.0.1:5000")
    socketio.run(app, debug=True, port=5000)