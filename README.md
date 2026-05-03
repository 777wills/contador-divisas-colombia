# Contador de Divisas Colombianas — YOLOv11

**Profesor: Ian Mateo Rodriguez Lopez**

**Estudiantes:**

- **William Alberto Suaza Losada**
- **Holman Sanchez**
- **Bairon Gutiérrez**

Sistema de visión artificial _end-to-end_ que detecta y cuantifica monedas y billetes colombianos en imágenes y video, basado en transfer learning con YOLOv11s. Construido como entregable para el curso de Proyecto Innovación II (Maestría en Inteligencia Artificial).

**Stack:** YOLOv11s + Ultralytics + Roboflow (etiquetado/augmentation) + ByteTrack (video estable).
**Entorno de ejecución híbrido:** Máquina local Ubuntu conectada a un entorno remito GPU (probado en A100/T4) mediante la extensión de Jupyter/Colab en VS Code, con persistencia vinculada a Google Drive.

**Métricas finales (run actual):**

| Split | Imágenes | Instancias | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
| ----- | -------- | ---------- | ------- | ------------ | --------- | ------ |
| Val   | 45       | 296        | 0.9741  | 0.8956       | 0.9195    | 0.9430 |
| Test  | 22       | 118        | 0.9739  | 0.9089       | 0.9585    | 0.9343 |

---

## 1. Estructura del repositorio

```
contador-divisas-colombia/
├── pipeline_monedas.ipynb              ← Notebook maestro (Entorno VS Code + Colab) — train/eval GPU
├── pipeline_monedas_edge_local.ipynb   ← Notebook edge local CPU (independiente, sin Drive)
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── samples/                        ← fotos locales + video_test.mp4 opcionales
│   └── dataset/                        ← (gitignored) dataset base extraído localmente
├── images/
│   └── robotflow/                      ← Gráficas documentales y soporte usadas en los notebooks
├── models/                             ← (gitignored) best_final.pt descargado de Drive
└── outputs/
    └── edge/                           ← (gitignored) salidas del notebook edge local
```

Toda la persistencia de entrenamiento de Colab está configurada en Google Drive bajo `MyDrive/DeteccionMonedas/`:

```
MyDrive/DeteccionMonedas/
├── dataset/
│   └── dataset_v1.tar.xz        ← input (re-exportado desde Roboflow)
├── samples/                      ← input — fotos nuevas + video_test.mp4
├── runs/
│   ├── monedas_v1/               ← logs y checkpoints del entrenamiento
│   └── monedas_v1_test/          ← validación en split test
├── models/
│   └── best_final.pt             ← modelo final
└── outputs/
    ├── inference_images/         ← fotos anotadas con sumatoria
    └── inference_videos/         ← video estabilizado + gráfica
```

---

## 2. Dataset y Preprocesamiento

**Recolección Original:** Se capturaron 225 fotografías originales abarcando denominaciones nuevas de monedas colombianas y billetes de $50.000, $20.000 y $10.000.

- **Condiciones y Oclusión:** Se aprovechó la luz natural y luz artificial focalizada para evitar falsos positivos por brillos en el metal. Las imágenes incluyen instancias aisladas, pero también oclusión y agrupaciones complejas en la misma foto.

**Partición y Aumentación (Roboflow):**
Previo a cualquier aumentación se realizó una división estricta 70/20/10 para evitar fuga de datos. Posteriormente se aplicó aumentación _offline_ ×3, quedando el dataset final distribuido de la siguiente forma:

- **Train:** 474 imágenes
- **Validation:** 45 imágenes
- **Test:** 22 imágenes

> _Nota:_ Adicional a las 22 imágenes de test, se incluyeron 27 muestras externas puras en `/data/samples/` para pruebas en frío.

---

## 3. Tarea cubierta

El entregable cumple las **4 tareas obligatorias** del enunciado y los **3 retos de bonificación**:

| #     | Tarea / Bonus                      | Sección del notebook                                       | Puntos      |
| ----- | ---------------------------------- | ---------------------------------------------------------- | ----------- |
| 1     | Captura de datos + iluminación     | Sección 1 (visualización del trabajo Roboflow)             | obligatoria |
| 2     | Anotación + preparación            | Roboflow GUI + Sección 1                                   | obligatoria |
| 3     | Entrenamiento (transfer learning)  | Sección 2 — justificación detallada de hiperparámetros     | obligatoria |
| 4     | Inferencia + sumatoria de dinero   | Sección 4 — formato literal de rúbrica                     | obligatoria |
| Bonus | Billetes (deformaciones complejas) | Integrado: 3 clases de billetes en el dataset              | +1          |
| Bonus | Edge computing local CPU           | Notebook independiente `pipeline_monedas_edge_local.ipynb` | +1          |
| Bonus | Video con conteo estable           | Sección 5 — ByteTrack + voto mayoritario + persistencia    | +2          |

---

## 4. Clases del modelo

```
moneda_50, moneda_100, moneda_200, moneda_500, moneda_1000,
billete_10000, billete_20000, billete_50000
```

Mapeo a valor monetario en pesos colombianos centralizado en `pipeline_monedas.ipynb` (Sección 0) y replicado en `pipeline_monedas_edge_local.ipynb` (autocontenido, sin Drive).

---

## 5. Cómo correrlo

El despliegue está bifurcado en dos rutas claras según la operatividad deseada: entrenamiento masivo con GPU virtual y Google Drive, o inferencia rápida local con CPU local pura.

### 5.1 Opción A: Entrenamiento vía VSCode+Colab (GPU)

Esta ruta utiliza el entorno de VS Code conectado a una sesión de Google Colab vía extensión, lo cual requiere mapear `Google Drive` para el entrenamiento pesado.

Abrir `pipeline_monedas.ipynb` en VS Code usando el túnel/kernel de Google Colab y ejecutar en el siguiente orden:

| Sección | Contenido                                                                           |
| ------- | ----------------------------------------------------------------------------------- |
| 0       | Setup: Drive, paths, `CLASS_VALUES`, descompresión, funciones reutilizables         |
| 1       | Exploración: tabla de clases × split, galería etiquetada, tamaños de caja, oclusión |
| 2       | Entrenamiento: justificación HP + `train_or_resume()` + sync a Drive                |
| 3       | Evaluación: `val()` en val y test, gráficas (results, confusión, PR, F1)            |
| 4       | Inferencia imagen: `quantify()` sobre test set + fotos nuevas                       |
| 5       | Inferencia video (BONUS +2pt): ByteTrack + `StableCounter` + métrica de estabilidad |
| 6       | Conclusiones técnicas: FP/FN, oclusión, limitaciones (referencia al notebook edge)  |

**Anti-desconexión:** El entrenamiento en esta rama está defendido contra posibles desconexiones de red, ya que el callback `make_drive_sync_callback` sincroniza pesos (`last.pt`, `best.pt`) a Drive al final de cada época.

---

### 5.2 Opción B: Bonus Edge local (CPU)

El bonus de despliegue edge se documenta en el notebook 100% separado `pipeline_monedas_edge_local.ipynb`. A diferencia del modo Colab, esta ruta corre **puramente nativa en su procesador local**, sin conectividad a servicios en la nube ni Drive. Utiliza un peso final `/models/best_final.pt` y aplica los mismos hiperparámetros lógicos (`imgsz=640`, `conf=0.4`, `iou=0.5`).

#### Setup local

```bash
# 1. Crear venv e instalar dependencias locales
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. El modelo best_final.pt ya debe estar presente en local: models/best_final.pt

# 3. Colocar fotos en data/samples/ y video corto en data/samples/video_test.mp4 (Opcional pero sugerido)
```

#### Ejecutar el notebook (CPU local)

Abrir `pipeline_monedas_edge_local.ipynb` seleccionando en VS Code explícitamente su kernel virtual `Python (.venv) local` (NO usar túnel Colab aquí). Estructura:

| Sección | Demo                                                                             | Métrica reportada      |
| ------- | -------------------------------------------------------------------------------- | ---------------------- |
| 0       | Setup local + constantes (`device='cpu'`, paths)                                 | —                      |
| 1       | Funciones core duplicadas (`format_currency`, `build_message`, `quantify_local`) | —                      |
| 2       | Cargar `models/best_final.pt`                                                    | clases del modelo      |
| 3       | Demo imagen única + galería 1×1 inline                                           | latencia (ms)          |
| 4       | Demo carpeta de imágenes → galería + tabla                                       | latencia promedio (ms) |
| 5       | Demo video archivo → escribe `outputs/edge/edge_video_v2.mp4` + 3 frames sample  | FPS CPU                |
| 6       | Demo webcam en vivo (`RUN_WEBCAM=False` por defecto)                             | FPS CPU                |
| 7       | Conclusiones edge + tabla consolidada de métricas                                | resumen                |

---

## 6. Justificación de hiperparámetros (resumen)

La Sección 2 del notebook desarrolla cada decisión. Resumen ejecutivo:

| Parámetro      | Valor                                                        | Razón                                                                         |
| -------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| Modelo         | `yolo11s.pt`                                                 | Balance precisión/velocidad para 8 clases × 474 imgs train (2 658 instancias) |
| Estrategia     | Fine-tune completo (sin freeze)                              | Dominio cercano a COCO + augmentation contra overfitting                      |
| Optimizer      | AdamW                                                        | Robusto en datasets pequeños; explícito en lugar de `auto`                    |
| `lr0`          | 0.01 + `cos_lr=True`                                         | Default Ultralytics + decay coseno suaviza al final                           |
| `epochs`       | 100 con `patience=20`                                        | Techo amplio + early-stop por meseta de mAP val                               |
| `batch`        | 16 con `imgsz=640`                                           | Compatible con T4 (16 GB) y A100 — `nbs=64` escala el LR                      |
| Augmentation   | **Reducido**: `mosaic=0.5`, `mixup=0`, `hsv_h=0`, `fliplr=0` | Roboflow ya hace augmentation offline ×3 — evitar double-augmentation         |
| `close_mosaic` | 10                                                           | Refina sobre imágenes "reales" en últimas 10 épocas                           |

---

## 7. Estabilizador de video Multi-Entorno (BONUS +2)

La Sección 5 implementa una pipeline de 3 capas para evitar el parpadeo del conteo entre frames. Debido a las latencias variables por hardware, se utilizan dos versiones diferentes ajustadas al entorno:

1. **Tracking persistente:** con ByteTrack (`model.track(persist=True)`).
2. **Persistencia mínima (Entorno GPU - V1):** Para el procesamiento GPU, un `track_id` solo cuenta si aparece en un estricto de ≥3 de los últimos 5 frames (`VIDEO_WINDOW = 5`, `VIDEO_MIN_HITS = 3`).
3. **Persistencia estabilizada (Entorno CPU Local - V2):** En un CPU con frames irregulares, la clase `StableCounterV2` exige ≥4 de los últimos 7 frames (`V2_WINDOW = 7`, `V2_MIN_HITS = 4`), más un filtro de _mediana móvil_ de 15 frames superpuesto para garantizar que el total global fluya naturalmente aun cuando caiga el target temporal.
4. **Voto mayoritario:** de clase por `track_id` (la clase asignada es la moda del historial).

Se grafica la comparativa **total crudo vs total estable** a lo largo del video — evidencia que los filtros reducen la magnitud y frecuencia de los cambios espurios.

---

## 8. Conclusiones técnicas

Las conclusiones reales (FP/FN, impacto de oclusión, limitaciones) se documentan en la Sección 6 del notebook tras correr el pipeline completo. Esa sección queda como entregable narrativo del proyecto e incluye una referencia al notebook edge local como entregable separado del bonus +1pt.
