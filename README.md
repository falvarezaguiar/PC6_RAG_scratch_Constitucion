# PC6 RAG from Scratch - Constitución Española

Pipeline RAG (Retrieval Augmented Generation) construido desde cero para consultar la Constitución Española mediante lenguaje natural.

## 📋 Descripción

Este proyecto implementa un sistema RAG completo que permite hacer preguntas sobre la **Constitución Española** y obtener respuestas generadas por un LLM con contexto relevante extraído del documento oficial.

### ¿Qué es RAG?

**RAG** = Retrieval Augmented Generation (Generación Aumentada por Recuperación)

Combina tres componentes:
- **Retrieval (Recuperación)**: Búsqueda semántica de información relevante
- **Augmented (Aumentado)**: Enriquecimiento del prompt con contexto recuperado
- **Generation (Generación)**: Respuesta generada por un LLM basada en contexto real

### Ventajas de RAG

1. **Previene alucinaciones**: El LLM genera respuestas basadas en hechos verificables del documento
2. **Datos personalizados**: Permite trabajar con documentos específicos no presentes en el entrenamiento del LLM
3. **Trazabilidad**: Acceso a las fuentes exactas de donde proviene cada respuesta
4. **Rápido de implementar**: Más ágil que hacer fine-tuning de un modelo

## 🎯 Casos de Uso

Este tipo de sistema RAG es ideal para:
- **Chatbots de documentación**: Q&A sobre manuales, normativas o documentación técnica
- **Asistentes legales**: Consultas sobre leyes, reglamentos y constituciones
- **Análisis de documentos**: Extracción de información estructurada de documentos largos
- **Soporte educativo**: Q&A sobre libros de texto y material de estudio

## 🏗️ Arquitectura del Pipeline

```
1. Carga de PDF → 2. Procesamiento de Texto → 3. Embeddings → 4. Búsqueda Vectorial → 5. Generación LLM
```

### Componentes Principales

1. **Procesamiento de Documentos**
   - Extracción de texto del PDF (PyMuPDF)
   - Filtrado por idioma (solo castellano)
   - División en chunks semánticos (spaCy)

2. **Embeddings**
   - Modelo: `all-mpnet-base-v2` (sentence-transformers)
   - Vectorización de chunks de texto
   - Almacenamiento en tensores PyTorch

3. **Búsqueda Semántica**
   - Embedding de consulta
   - Búsqueda por similitud de coseno (dot product)
   - Recuperación de top-k chunks más relevantes

4. **Generación de Respuestas**
   - LLM: Gemma-2B-Instruct (Google/Keras)
   - Prompt engineering con contexto aumentado
   - Respuestas en español basadas en artículos constitucionales

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| **Entorno** | Google Colab (GPU) |
| **Procesamiento PDF** | PyMuPDF (fitz) |
| **NLP** | spaCy (`es_core_news_sm`) |
| **Embeddings** | sentence-transformers |
| **Framework DL** | PyTorch, Keras-NLP |
| **LLM** | Gemma-2B-Instruct (JAX backend) |
| **Análisis** | pandas, numpy |

## 📦 Instalación

### Dependencias principales

```bash
pip install PyMuPDF tqdm
pip install spacy
python -m spacy download es_core_news_sm
pip install sentence-transformers
pip install -U keras-nlp keras>=3
```

### Configuración en Google Colab

```python
# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Configurar credenciales de Kaggle (para Gemma)
from google.colab import userdata
import os
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

## 🚀 Uso

### 1. Estructura de Datos

El documento debe estar en Google Drive:
```
/content/drive/MyDrive/data/BOE-151_Constitucion_Espanola.pdf
```

### 2. Ejecución del Notebook

Ejecutar las celdas secuencialmente:

1. **Carga del PDF**: Monta Drive y copia el archivo localmente
2. **Extracción de texto**: Procesa el PDF y filtra páginas en castellano (0-36)
3. **Chunking**: Divide en sentencias con spaCy, agrupa en chunks de ~10 frases
4. **Embeddings**: Genera vectores con `all-mpnet-base-v2`
5. **Búsqueda**: Implementa función de recuperación por similitud
6. **Generación**: Carga Gemma-2B y genera respuestas con contexto

### 3. Ejemplo de Consulta

```python
query = "¿Cuáles son las funciones del Gobierno?"

# Recuperar contexto relevante
scores, indices = retrieve_relevant_resources(query, n_resources_to_return=5)

# Construir prompt aumentado
context_items = [pages_and_chunks[i]["sentence_chunk"] for i in indices]
prompt = f"""Basado en el siguiente contexto de la Constitución Española:
{chr(10).join([f"- {item}" for item in context_items])}

Pregunta: {query}

Respuesta:"""

# Generar respuesta
response = gemma_lm.generate(prompt, max_length=256)
print(response)
```

## 📊 Procesamiento de Datos

### Estadísticas del Documento

- **Páginas procesadas**: 37 (versión castellano)
- **Total chunks**: ~300-400 (variable según configuración)
- **Tamaño medio chunk**: ~200-500 caracteres
- **Tokens por chunk**: ~50-125 tokens (aprox)

### Pipeline de Chunking

```
Páginas completas → Sentencias (spaCy) → Agrupación (10 sentencias) → Filtrado (>30 tokens)
```

## 🧪 Comparación: Sin RAG vs Con RAG

### Sin RAG (Solo conocimiento del modelo)
```
Query: "¿Es posible tener doble nacionalidad española?"
Respuesta: "Sí, es posible... [respuesta genérica basada en conocimiento general]"
⚠️ Puede ser imprecisa o desactualizada
```

### Con RAG (Con contexto constitucional)
```
Query: "¿Es posible tener doble nacionalidad española?"
Respuesta: "Según el Artículo 11.3 de la Constitución Española: 
'El Estado podrá concertar tratados de doble nacionalidad con los países 
iberoamericanos o con aquellos que hayan tenido o tengan una particular 
vinculación con España...'"
✅ Respuesta precisa con fuente verificable
```

## 📁 Estructura del Proyecto

```
PC6/
├── PC6_rag-scratch-Constitucion.ipynb  # Notebook principal
├── PC6_rag-scratch-Constitucion.pdf    # PDF exportado del notebook
├── README.md                            # Este archivo
├── data/
│   └── BOE-151_Constitucion_Espanola.pdf
├── @comands/
│   └── PromptDisipliando.md
└── kaggle.json                          # Credenciales Kaggle (no commitear)
```

## 🔑 Configuración de Kaggle

Para usar Gemma-2B necesitas credenciales de Kaggle:

1. Ir a https://www.kaggle.com/settings/account
2. Crear un nuevo token API
3. Descargar `kaggle.json`
4. En Colab: Secrets → Añadir `KAGGLE_USERNAME` y `KAGGLE_KEY`

## ⚙️ Optimizaciones Implementadas

- **Filtrado de idioma**: Solo procesa páginas en castellano
- **Chunking adaptativo**: Agrupa sentencias para contexto óptimo
- **Filtrado de tokens**: Elimina chunks muy cortos (<30 tokens)
- **GPU acceleration**: Embeddings y generación en CUDA
- **Caché local**: Copia PDF a Colab para procesamiento rápido

## 🎓 Conceptos Clave Implementados

- **Embeddings semánticos**: Representación vectorial de texto
- **Similitud de coseno**: Búsqueda por dot product de vectores
- **Prompt engineering**: Diseño de prompts con contexto estructurado
- **Chunking estratégico**: Balance entre contexto y granularidad
- **Text preprocessing**: Limpieza y normalización de texto

## 📝 Notas de Desarrollo

- El modelo spaCy puede requerir reinicio del kernel tras instalación
- La primera carga de Gemma-2B descarga ~5GB de pesos
- Usar backend JAX (`KERAS_BACKEND="jax"`) para mejor rendimiento
- Los chunks muy cortos se filtran para evitar ruido en búsqueda
- La Constitución incluye 5 idiomas, solo se procesa castellano

## 🤝 Contribuciones

Este proyecto es parte del **Master en AI** - Práctica Computacional 6 (PC6).

Para contribuir:
1. Fork del repositorio
2. Crear branch de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Añade nueva funcionalidad'`)
4. Push al branch (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto es material educativo. El documento de la Constitución Española es de dominio público (BOE).

## 🔗 Referencias

- [Constitución Española (BOE)](https://www.boe.es/buscar/act.php?id=BOE-A-1978-31229)
- [RAG Paper (2020)](https://arxiv.org/abs/2005.11401)
- [Sentence Transformers](https://www.sbert.net/)
- [Gemma Models (Google)](https://ai.google.dev/gemma)
- [spaCy Documentation](https://spacy.io/)

## 👨‍💻 Autor

Desarrollado como parte del programa de Master en Inteligencia Artificial.

---

**Última actualización**: Noviembre 2025


