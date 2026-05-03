# Contador de Divisas Colombianas — Pipeline YOLOv11

Sistema de visión artificial _end-to-end_ que detecta y cuantifica monedas y billetes colombianos en imágenes y video, basado en transfer learning con YOLOv11s. Construido como entregable para el curso de Proyecto Innovación II (Maestría en Inteligencia Artificial).

**Stack:** YOLOv11s + Ultralytics + Roboflow (etiquetado/augmentation) + Google Colab (probado en A100; compatible con T4) + ByteTrack (video estable).

**Métricas finales (run actual):**

| Split | Imágenes | Instancias | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|---|---|
| Val  | 45 | 296 | 0.9741 | 0.8956 | 0.9195 | 0.9430 |
| Test | 22 | 118 | 0.9739 | 0.9089 | 0.9585 | 0.9343 |

---

## 1. Estructura del repositorio

```
contador-divisas-colombia/
├── pipeline_monedas.ipynb              ← Notebook maestro (Colab) — train/eval/inferencia GPU
├── pipeline_monedas_edge_local.ipynb   ← Notebook edge local CPU (independiente, sin Drive)
├── ROBOFLOW_INSTRUCTIONS.md            ← Cómo regenerar el dataset en Roboflow
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── samples/                        ← (vacío) fotos locales + video_test.mp4 opcionales
│   └── dataset/                        ← (gitignored) dataset descargado para uso local
├── models/                             ← (gitignored) best_final.pt descargado de Drive
└── outputs/
    ├── inference_images/               ← (gitignored) salidas Colab — fotos anotadas
    ├── inference_videos/               ← (gitignored) salidas Colab — video estabilizado
    └── edge/                           ← (gitignored) salidas del notebook edge local
```

Toda la persistencia pesada está en Google Drive bajo `MyDrive/DeteccionMonedas/`:

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

## 2. Tarea cubierta

El entregable cumple las **4 tareas obligatorias** del enunciado y los **3 retos de bonificación**:

| # | Tarea / Bonus | Sección del notebook | Puntos |
|---|---|---|---|
| 1 | Captura de datos + iluminación | Sección 1 (visualización del trabajo Roboflow) | obligatoria |
| 2 | Anotación + preparación | Roboflow GUI + Sección 1 | obligatoria |
| 3 | Entrenamiento (transfer learning) | Sección 2 — justificación detallada de hiperparámetros | obligatoria |
| 4 | Inferencia + sumatoria de dinero | Sección 4 — formato literal de rúbrica | obligatoria |
| Bonus | Billetes (deformaciones complejas) | Integrado: 3 clases de billetes en el dataset | +1 |
| Bonus | Edge computing local CPU | Notebook independiente `pipeline_monedas_edge_local.ipynb` | +1 |
| Bonus | Video con conteo estable | Sección 5 — ByteTrack + voto mayoritario + persistencia | +2 |

---

## 3. Clases del modelo

```
moneda_50, moneda_100, moneda_200, moneda_500, moneda_1000,
billete_10000, billete_20000, billete_50000
```

Mapeo a valor monetario en pesos colombianos centralizado en `pipeline_monedas.ipynb` (Sección 0) y replicado en `pipeline_monedas_edge_local.ipynb` (autocontenido, sin Drive).

---

## 4. Cómo correrlo

### 4.1 Pre-requisitos (Colab)

1. **Generar el dataset desde Roboflow** siguiendo [`ROBOFLOW_INSTRUCTIONS.md`](ROBOFLOW_INSTRUCTIONS.md). Sube el `.tar.xz` a `MyDrive/DeteccionMonedas/dataset/dataset_v1.tar.xz`.
2. **Capturar imágenes nuevas** (5-10 fotos del celular) y subirlas a `MyDrive/DeteccionMonedas/samples/`.
3. **Grabar un video** de 15-30 segundos pasando piezas frente a la cámara y subirlo a `MyDrive/DeteccionMonedas/samples/video_test.mp4`.
4. **Activar GPU T4** en Colab: `Runtime → Change runtime type → T4 GPU`.

### 4.2 Ejecutar el notebook

Abrir `pipeline_monedas.ipynb` en Colab (o desde VS Code con la extensión de Colab) y ejecutar las celdas en orden. Cada sección tiene markdown explicativo:

| Sección | Contenido |
|---|---|
| 0 | Setup: Drive, paths, `CLASS_VALUES`, descompresión, funciones reutilizables |
| 1 | Exploración: tabla de clases × split, galería etiquetada, tamaños de caja, oclusión |
| 2 | Entrenamiento: justificación HP + `train_or_resume()` + sync a Drive |
| 3 | Evaluación: `val()` en val y test, gráficas (results, confusión, PR, F1) |
| 4 | Inferencia imagen: `quantify()` sobre test set + fotos nuevas |
| 5 | Inferencia video (BONUS +2pt): ByteTrack + `StableCounter` + métrica de estabilidad |
| 6 | Conclusiones técnicas: FP/FN, oclusión, limitaciones (referencia al notebook edge) |

### 4.3 Anti-desconexión Colab

El entrenamiento (Sección 2) registra un callback `on_fit_epoch_end` que sincroniza `last.pt`, `best.pt`, `results.csv` y `args.yaml` a Drive **al final de cada época**. Si Colab desconecta, simplemente reconecta y vuelve a ejecutar todas las celdas: `train_or_resume()` detectará el checkpoint en Drive y reanudará con `resume=True`.

---

## 5. Bonus: Edge local (CPU)

El bonus de despliegue edge se entrega en `pipeline_monedas_edge_local.ipynb`,
un notebook **independiente** del maestro que ejecuta inferencia 100% local sobre
CPU sin Colab y sin Google Drive. Usa los mismos pesos `models/best_final.pt`
y la misma config (`imgsz=640`, `conf=0.4`, `iou=0.5`) — esto garantiza que la
métrica reportada es comparable contra el throughput GPU del notebook maestro.

### 5.1 Setup local

```bash
# 1. Crear venv e instalar dependencias
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Descargar best_final.pt desde MyDrive/DeteccionMonedas/models/
#    y colocarlo en models/best_final.pt

# 3. (Opcional) Colocar fotos en data/samples/ y un video corto en
#    data/samples/video_test.mp4 para las demos 3-5
```

### 5.2 Ejecutar el notebook

Abrir `pipeline_monedas_edge_local.ipynb` en VS Code con kernel **`Python (.venv)`
local** (NO el kernel Colab) y ejecutar `Run All`. Estructura interna:

| Sección | Demo | Métrica reportada |
|---|---|---|
| 0 | Setup local + constantes (`device='cpu'`, paths) | — |
| 1 | Funciones core duplicadas (`format_currency`, `build_message`, `quantify_local`) | — |
| 2 | Cargar `models/best_final.pt` | clases del modelo |
| 3 | Demo imagen única + galería 1×1 inline | latencia (ms) |
| 4 | Demo carpeta de imágenes → galería + tabla | latencia promedio (ms) |
| 5 | Demo video archivo → escribe `outputs/edge/edge_video_v2.mp4` + 3 frames sample | FPS CPU |
| 6 | Demo webcam en vivo (`RUN_WEBCAM=False` por defecto) | FPS CPU |
| 7 | Conclusiones edge + tabla consolidada de métricas | resumen |

### 5.3 Notas de uso

- El notebook usa **`StableCounterV2`** (mismo estabilizador que la Sección 5 del
  maestro, con mediana móvil) para video y webcam. El conteo es idéntico al de
  Colab; lo que cambia es solo el throughput.
- La **celda de webcam está deshabilitada por defecto** para que `Run All` termine
  sin colgarse. Para activarla cambia `RUN_WEBCAM = True` en la celda 2 y
  re-ejecuta solo esa celda. Las instrucciones detalladas (requisitos de display
  X11, fallback sin GUI, tope de frames, tecla `q` para salir) están dentro del
  propio notebook.
- Salidas escritas a `outputs/edge/` (gitignored): `pred_single_*.jpg`,
  `pred_folder_*.jpg`, `edge_video_v2.mp4`, `edge_webcam_v2.mp4`.

---

## 6. Justificación de hiperparámetros (resumen)

La Sección 2 del notebook desarrolla cada decisión. Resumen ejecutivo:

| Parámetro | Valor | Razón |
|---|---|---|
| Modelo | `yolo11s.pt` | Balance precisión/velocidad para 8 clases × 474 imgs train (2 658 instancias) |
| Estrategia | Fine-tune completo (sin freeze) | Dominio cercano a COCO + augmentation contra overfitting |
| Optimizer | AdamW | Robusto en datasets pequeños; explícito en lugar de `auto` |
| `lr0` | 0.01 + `cos_lr=True` | Default Ultralytics + decay coseno suaviza al final |
| `epochs` | 100 con `patience=20` | Techo amplio + early-stop por meseta de mAP val |
| `batch` | 16 con `imgsz=640` | Compatible con T4 (16 GB) y A100 — `nbs=64` escala el LR |
| Augmentation | **Reducido**: `mosaic=0.5`, `mixup=0`, `hsv_h=0`, `fliplr=0` | Roboflow ya hace augmentation offline ×3 — evitar double-augmentation |
| `close_mosaic` | 10 | Refina sobre imágenes "reales" en últimas 10 épocas |

---

## 7. Estabilizador de video (BONUS +2)

La Sección 5 implementa una pipeline de 3 capas para evitar el parpadeo del conteo entre frames:

1. **Tracking persistente** con ByteTrack (`model.track(persist=True)`).
2. **Persistencia mínima:** un `track_id` solo cuenta si aparece en ≥3 de los últimos 5 frames.
3. **Voto mayoritario** de clase por `track_id` (la clase asignada es la moda del historial, no la del frame actual).

Se grafica la comparativa **total crudo vs total estable** a lo largo del video — evidencia que el filtro reduce la magnitud y frecuencia de los cambios espurios.

---

## 8. Conclusiones técnicas

Las conclusiones reales (FP/FN, impacto de oclusión, limitaciones) se documentan en la Sección 6 del notebook tras correr el pipeline completo. Esa sección queda como entregable narrativo del proyecto e incluye una referencia al notebook edge local como entregable separado del bonus +1pt.
