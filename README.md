# Detección de Objetos con Teachable Machine

Clasificador de imágenes en tiempo real que usa un modelo entrenado en [Google Teachable Machine](https://teachablemachine.withgoogle.com/) para detectar, a través de la cámara web, si un objeto específico está presente o no en la escena.

El modelo distingue dos clases:

- **el objeto**: el objeto entrenado está frente a la cámara.
- **sin objeto**: la escena está vacía (o no se detecta el objeto).

## Requisitos

- Python 3.10 o superior
- Una cámara web conectada
- Windows, macOS o Linux

## Instalación

1. Clona el repositorio:

   ```bash
   git clone https://github.com/Falzo186/deteccion-de-objetos.git
   cd deteccion-de-objetos
   ```

2. (Recomendado) Crea un entorno virtual:

   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## Uso

Ejecuta el script principal:

```bash
python detectar_objeto.py
```

Se abrirá una ventana con la imagen de la cámara mostrando la clase detectada y el porcentaje de confianza en tiempo real.

Presiona **`q`** con la ventana de la cámara en foco para cerrar el programa.

## Estructura del proyecto

```
deteccion-de-objetos/
├── detectar_objeto.py      # Script principal de inferencia en tiempo real
├── requirements.txt        # Dependencias de Python
├── Models/
│   ├── model_unquant.tflite  # Modelo exportado desde Teachable Machine (TensorFlow Lite, Floating point)
│   └── labels.txt             # Etiquetas de las clases
└── README.md
```

## Notas

- El modelo fue entrenado con 50 imágenes de la clase "el objeto" y 50 imágenes de la clase "sin objeto" usando Teachable Machine.
- Las rutas del modelo y las etiquetas son relativas a la ubicación del script, por lo que el proyecto funciona igual sin importar en qué carpeta o equipo se clone.
- Si tienes más de una cámara conectada, cambia `cv2.VideoCapture(0)` por el índice correspondiente (`1`, `2`, ...) en `detectar_objeto.py`.
