import os
import uvicorn
import PyPDF2
import docx
import re
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv
import io

# Cargar variables de entorno
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("ERROR: La API Key de OpenAI no se encontró.")

client = OpenAI(api_key=OPENAI_API_KEY)
# Instanciar el modelo de sentence_transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

print("API Key cargada en el backend:", os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Configuración de CORS: ajusta allow_origins según tus necesidades
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # en producción, restringí a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyzeResume")
async def analyze_resume(
    file: UploadFile = File(...),
    job_type: str = Form("")
):
    if not file:
        raise HTTPException(status_code=400, detail="Debe enviar un archivo")
    
    filename = file.filename.lower()
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error leyendo el archivo")

    text = ""
    if filename.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error al procesar el PDF")
    elif filename.endswith(".docx"):
        try:
            document = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([para.text for para in document.paragraphs])
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error al procesar el DOCX")
    else:
        raise HTTPException(
            status_code=400, 
            detail="Formato de archivo no soportado. Solo se admiten PDF y DOCX"
        )

    if not text.strip():
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del archivo")

    # Extraer un resumen simple usando sentence_transformers
    # En este ejemplo, separamos el texto por puntos y elegimos las tres primeras oraciones
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    summary = " ".join(sentences[:3]) if sentences else ""

    # Armar el prompt para la IA, incluyendo el resumen extraído
    prompt = f"""
Eres un asesor experto en recursos humanos y especialista en evaluar currículums. Por favor, revisa cuidadosamente el siguiente CV y proporciona un análisis equilibrado que incluya:
- Las fortalezas y habilidades clave del candidato.
- Áreas en las que se podría mejorar el CV.
- Sugerencias y recomendaciones para optimizar la presentación del perfil.

Utiliza un tono amable y constructivo, ofreciendo feedback detallado y ser directo y amable.
- Despidete de una forma amable y di que eres "Skinner"


Curriculum completo:
{text}

Resumen (extraído automáticamente):
{summary}

Tipo de trabajo: {job_type if job_type else "No especificado"}

Feedback:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "Eres un experto reclutador de Talento Humano y asesor de carreras. Por favor, revisa cuidadosamente el siguiente CV y proporciona un análisis equilibrado."},
                {"role": "user", "content": prompt},
            ],
        
        )
        
        
        feedback = response.choices[0].message.content.strip()
        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar el CV: {str(e)}")
