import cv2
import mediapipe as mp
import time

def pilih_kamera():
    """Mencari daftar kamera yang terhubung ke perangkat dan memilih salah satu."""
    kamera_tersedia = []
    print("Sedang mencari kamera yang terhubung...")
    
    # Memeriksa indeks 0 sampai 4
    for index in range(5):
        cap_test = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap_test.isOpened():
            kamera_tersedia.append(index)
            cap_test.release()
            
    if not kamera_tersedia:
        print("Peringatan: Tidak ada kamera yang terdeteksi!")
        return 0  # Default ke 0 jika gagal
    
    print(f"\nKamera terdeteksi di indeks: {kamera_tersedia}")
    
    # Jika ada lebih dari 1 kamera (misal: internal [0] & webcam USB [1])
    if len(kamera_tersedia) > 1:
        # Pilihan otomatis: Menggunakan webcam eksternal (indeks tertinggi)
        kamera_terpilih = kamera_tersedia[-1] 
        print(f"Otomatis memilih Webcam Eksternal (Indeks {kamera_terpilih}).")
        return kamera_terpilih
    else:
        print(f"Menggunakan kamera internal (Indeks {kamera_tersedia[0]}).")
        return kamera_tersedia[0]


# --- 1. INISIALISASI KAMERA --- jika pake webcam
CAM_INDEX = pilih_kamera()
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)


# --- 2. INISIALISASI MEDIAPIPE POSE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,           # Ubah ke 2 untuk akurasi tertinggi
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Kustomisasi warna & gaya garis visualisasi
landmark_drawing_spec = mp_drawing.DrawingSpec(
    color=(0, 255, 0),       # Warna titik (Hijau - BGR)
    thickness=2, 
    circle_radius=4
)
connection_drawing_spec = mp_drawing.DrawingSpec(
    color=(255, 100, 0),     # Warna garis (Oranye - BGR)
    thickness=2
)

# Variabel perhitungan FPS
prev_time = 0

print(f"\nKamera terhubung (Indeks {CAM_INDEX})! Tekan 'q' pada jendela video untuk keluar.")


# --- 3. LOOP UTAMA ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Gagal mengambil frame dari kamera.")
        break

    # Konversi BGR ke RGB untuk proses MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Drawing landmarks jika pose terdeteksi
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec
        )
        
        # Contoh mengambil koordinat hidung (Nose)
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        h, w, _ = frame.shape
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        
        cv2.putText(frame, "Tubuh Terdeteksi", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Mencari Tubuh...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Perhitungan FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    
    # Overlay FPS & Info Kamera pada Layar
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Cam Index: {CAM_INDEX}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("MediaPipe Pose - Custom Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
pose.close()