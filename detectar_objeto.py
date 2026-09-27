from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# --- Configuración ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Models" / "model_unquant.tflite"
LABELS_PATH = BASE_DIR / "Models" / "labels.txt"

# --- Cargar modelo y etiquetas ---
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    class_names = [
        line.strip().split(" ", 1)[1]
        for line in file
        if line.strip()
    ]

print("Entrada:", input_info["shape"], input_info["dtype"])
print("Salida:", output_info["shape"], output_info["dtype"])
print("Clases:", class_names)


def prepare_image(frame):
    """Prepara el cuadro en el formato esperado por Teachable Machine."""
    height, width = frame.shape[:2]
    side = min(height, width)

    x = (width - side) // 2
    y = (height - side) // 2
    square = frame[y:y + side, x:x + side]

    input_height = int(input_info["shape"][1])
    input_width = int(input_info["shape"][2])

    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_width, input_height))

    normalized = (resized.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized, axis=0)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = prepare_image(frame)

        interpreter.set_tensor(input_info["index"], image)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_info["index"])[0]

        index = int(np.argmax(predictions))
        name = class_names[index]
        confidence = float(predictions[index])

        cv2.putText(frame, f"{name}: {confidence:.1%}", (30, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 2,)

        cv2.imshow("Clasificador", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
