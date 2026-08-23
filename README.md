# MediVision
Website berbasis AI untuk membantu mendeteksi indikasi gangguan kesehatan melalui analisis tubuh secara visual.

tutorial instal media pipe (yang belum berbasis web )

buat file python baru contoh : "test.py"
lalu buka terminal ctrl + ` 

lalu masukan perintah ini di terminal 
"pip install mediapipe opencv-python"

tunggu sampai instalasi selesai 
,setelah selesai menginstal copy kode yang udah aku kirim 

kalo masih belum bisa jalan restart vs code kalian 

di dalem kode itu ada untuk deteksi kamera udah ku buatin command nya jadi kalian ganti kode index nya jadi kayak gini

# --- 1. INISIALISASI KAMERA ---
CAM_INDEX = 0  # 0 adalah indeks standar untuk kamera internal laptop
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
