# Transfer Learning con Autoencoder – Fashion MNIST

## Objetivo

Aplicar transfer learning en dos etapas:

Etapa 1:
Entrenamiento de un autoencoder convolucional para extracción no supervisada de características.

Etapa 2:
Uso del encoder entrenado como extractor fijo de características y entrenamiento de una cabeza clasificadora supervisada.

---

## Dataset

Fashion MNIST

60 000 imágenes de entrenamiento
10 000 imágenes de prueba
Resolución: 28 × 28
Clases: 10

---

## Arquitectura

### Encoder

Entrada: (28, 28, 1)

Conv2D(32, stride=2)
Conv2D(64, stride=2)
Flatten
Dense(128)
Latente: ( z \in \mathbb{R}^{64} )

### Decoder

Dense(7×7×64)
Conv2DTranspose(64)
Conv2DTranspose(32)
Conv2D(1, sigmoid)

### Clasificador

Encoder congelado
Dense(128)
Dropout(0.3)
Dense(10, softmax)

---

## Flujo de Ramas (MLOps básica)

master
etapa1 – entrenamiento del autoencoder
etapa2 – transferencia y clasificación

El modelo final se encuentra en la rama master después del merge de etapa2.

---

## Ejecución

Activar entorno:

```bash
conda activate fmnist_transfer
```

Etapa 1:

```bash
python src/train_stage1_autoencoder.py
```

Etapa 2:

```bash
python src/train_stage2_classifier.py
```

---

## Formulación Matemática

Sea

$$f_\theta : \mathbb{R}^{784} \rightarrow \mathbb{R}^{64}$$


el encoder entrenado en etapa 1.

La etapa 2 implementa:


$$h(x) = g_\phi(f_\theta(x))$$


donde $$ f_\theta )$$ permanece congelado y solo se optimiza $$ \phi )$$.

