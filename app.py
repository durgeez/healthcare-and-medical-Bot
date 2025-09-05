# chatbot.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from openai import OpenAI
import os, re
import random

app = FastAPI()

# Allow Storyline/browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === Keywords (Healthcare & Medical terms) ===
KEYWORDS = [
    "Management challenge", "Social imperative", "Multi-disciplinary collaboration", "Leadership development", "Demographic shifts",
    "Aging of population", "Life expectancy", "Chronic disease", "Interplay of Sectors", "Challenges and Opportunities",
    "HBS Healthcare Initiative", "Biotech and Pharma Panel", "Healthcare VC/Entrepreneurship Panel",
    "Payor and Provider Panel", "Devices and Diagnostics Panel", "Alumni mentor program", "Computer systems",
    "IT solutions", "Technology vendors", "Personal genetic information services", "Retail and workplace clinics",
    "Biologics", "Pharmaceuticals", "Devices", "Diagnostics", "Durable medical equipment",
    "Consumer health and wellness", "Over-the-counter drugs", "Diet and nutrition", "Vitamins and supplements",
    "Meal-delivery services", "Fitness and exercise", "Telemedicine", "Mr. Amarkumar M Raval",
    "Mrs. Fayejabanu S. Zankhwala", "Pharmarocks Medicine", "diseases and disorders", "professional medical advice",
    "self-medication", "pharmacy students", "medical students", "general practitioners", "health care professionals",
    "Introduction of disease", "Causes of disease", "Symptoms of disease", "Diagnosis of disease",
    "Treatment & medicine options", "125 diseases", "351 pages", "Acne", "Acute Hepatic Porphyria", "Allergy",
    "Amenorrhea", "Anemia", "Angina Pectoris", "Anxiety", "Arrythmia", "Arthritis", "Asthma", "Bell's Palsy",
    "Blood Sugar", "Boils", "Breast Cancer", "Breast Enlargement", "Burning feet", "Burning", "Cancer",
    "Cervical Dysplasia", "Chikungunya", "Congestive Heart Failure", "Conjuctivitis", "Constipation", "Cystitis",
    "Dehydration", "Dengue Fever", "Dental Infection", "Depression", "Dry Cough", "Dysentry", "Eczema",
    "Erectile Dysfunction", "Fatty Liver", "Fever", "Gall Bladder Stone", "Gas Trouble", "Gastritis", "Giardiasis",
    "Glaucoma", "H.Pylori Infection", "Hair Fall", "Head Ache", "Heat Stroke", "Hemophilia", "Hepatitis", "HFMD",
    "Hiccups", "Histeria", "HIV", "Hydrocele", "Hyperhidrosis", "Hypertension", "Hypotension", "Hypothermia",
    "Immunoglobulin-E", "Inflammation", "Inflammatory Bowel Disease", "Insomnia", "Itching", "Jaundice",
    "Joint Pain", "Keloid Scars", "Kidney Stone", "Leprosy", "Leishmaniasis", "Low Platelet Count",
    "Low WBC Count", "Meningitis", "Menorrhagia", "Migraine", "Morning Sickness", "Motion Sickness",
    "Mouth Ulcer", "Mucormycosis", "Malaria", "NAFLD", "Nausea", "Nerve Pain", "Night Fall", "Obesity",
    "Ocular Hypertension", "Oral Thrush", "Ovarian Cyst", "Paralysis", "Pelvic Pain", "PID", "Penile Papules",
    "Penis Yeast Infection", "Peptic Ulcer", "Phimosis", "Pneumonia", "PCOD", "Pus Formation",
    "Respiratory Tract Disease", "Rheumatoid Arthritis", "Ringworm", "Schizophrenia", "Scurvy", "Skin Diseases",
    "Scabies", "Splenomegaly", "Thyroid", "Tooth Pain", "Tremor", "Tuberculosis", "Typhoid Fever",
    "Urine Infection", "Vaginal Discharge", "Vaginal Dryness", "Vaginal Infection", "Varicocele", "Vertigo",
    "Viral Fever", "Viral Infection", "Vomitting", "Vulvovaginitis", "Weakness and Tiredness", "Wet Cough",
    "Worm Infection", "Ear Infection", "Chronic Kidney Disease", "Piles", "Psoriasis", "Chickenpox", "Vitiligo",
    "OD", "BID", "TID", "QID", "IM", "IV", "MRI", "CT Scan", "X-RAY", "Rx", "Blood Test", "Urine Test",
    "Genetic Testing", "Physical Check Up", "Medical History", "Electrocardiogram (ECG)", "Echocardiogram",
    "Spirometer test", "Electromyography (EMG)", "Ultrasound", "Biopsy", "Antihistamines", "Corticosteroids",
    "Antibiotics", "Antifungal", "Antivirals", "Homeopathy", "Ayurvedic", "Herbal medicine", "Talk therapy",
    "Meditation", "Yoga", "Surgery", "Painkillers", "Analgesic", "Antipyretic", "Inflammation", "Fatigue",
    "Weakness", "Fever", "Pain", "Swelling", "Itching", "Nausea", "Vomiting", "Headache", "Abdominal pain",
    "High blood pressure", "Low blood pressure", "Hormonal imbalance", "Nutritional deficiency",
    "Genetic mutation", "Stress", "Anxiety", "Obesity", "Infection", "Contaminated food or water", "Allergen",
    "Immune system", "Blood", "Liver", "Heart", "Kidney", "Lungs", "Nervous system", "Reproductive system",
    "Hair follicles", "Skin", "Joints"
]

# === Load PDFs ===
# Note: The original PDF paths are assumed to contain relevant medical content.
PDF_PATH = os.environ.get("PDF_PATH", "Healthcare_basics.pdf")
PDF_PATH_2 = os.environ.get("PDF_PATH_2", "diseases_and_treatments.pdf")

def load_pdf_text(path):
    if not os.path.exists(path):
        return ""
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        if txt.strip():
            pages.append(txt.strip())
    return "\n\n".join(pages)

PDF_TEXT = load_pdf_text(PDF_PATH) + "\n\n" + load_pdf_text(PDF_PATH_2)

# === Context matching functions ===
def find_relevant_context_keywords(question: str, pdf_text: str, top_k: int = 5):
    q_low = question.lower()
    matched_keywords = [kw for kw in KEYWORDS if kw.lower() in q_low]
    potential_contexts = []
    for para in pdf_text.split("\n\n"):
        score = sum(1 for kw in matched_keywords if kw.lower() in para.lower())
        if score > 0:
            potential_contexts.append((score, para))
    potential_contexts.sort(key=lambda x: x[0], reverse=True)
    return [p[1] for p in potential_contexts[:top_k]]

def find_relevant_context_semantic(question: str, pdf_text: str, top_k: int = 5):
    words = [w for w in question.lower().split() if len(w) > 2]
    potential_contexts = []
    for para in pdf_text.split("\n\n"):
        score = sum(1 for w in words if w in para.lower())
        if score > 0:
            potential_contexts.append((score, para))
    potential_contexts.sort(key=lambda x: x[0], reverse=True)
    return [p[1] for p in potential_contexts[:top_k]]

# === Image generation (keyword + semantic fallback) ===
async def generate_image(user_prompt: str, image_size: str):
    context = find_relevant_context_keywords(user_prompt, PDF_TEXT, top_k=3)
    if not context:
        context = find_relevant_context_semantic(user_prompt, PDF_TEXT, top_k=3)
    if not context:
        return JSONResponse({"answer": "I don't have enough information to generate an image related to that topic."})

    relevant_context = "\n\n".join(context)
    if image_size not in ["256x256", "512x512", "1024x1024"]:
        image_size = "512x512"

    try:
        img_resp = client.images.generate(
            model="dall-e-2",
            prompt=f"Generate an educational diagram strictly related to this healthcare and medical PDF context:\n{relevant_context}\n\nUser request: {user_prompt}",
            size=image_size
        )
        image_url = img_resp.data[0].url
        return JSONResponse({
            "answer": f"Here is an image based on the PDF context (size: {image_size}):",
            "image_url": image_url,
            "size": image_size
        })
    except Exception as e:
        print(f"[ERROR] Image generation failed: {e}")
        return JSONResponse({"answer": "Sorry, I couldn't generate an image right now."})

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    image_size = data.get("image_size", "512x512")

    if not question:
        return JSONResponse({"answer": "Please enter a question."})

    # Handle greetings directly without emojis
    greetings = ["hi", "hello", "good morning", "good afternoon", "good evening"]
    if any(re.search(r'\b' + re.escape(g) + r'\b', question.lower()) for g in greetings):
        response_options = [
            "Hello! I can provide information on healthcare and medical topics. How can I assist you?",
            "Hi there! I'm a chatbot specializing in medical knowledge. What would you like to know?",
            "Greetings! Feel free to ask me questions about diseases, treatments, or healthcare concepts.",
        ]
        return JSONResponse({"answer": random.choice(response_options)})

    # Detect image request
    if re.search(r"\b(generate|create|show|make)\b.*\b(image|picture|diagram|visual)\b", question.lower()):
        return await generate_image(question, image_size)

    # Use both keyword and semantic context
    context = find_relevant_context_keywords(question, PDF_TEXT)
    if not context:
        context = find_relevant_context_semantic(question, PDF_TEXT)

    relevant_context = "\n\n".join(context)

    try:
        # Updated system prompt for conversational tone and broader knowledge
        system_prompt = (
            "You are a healthcare and medical knowledge expert chatbot. "
            "You will only answer questions about healthcare, diseases, treatments, and related topics. "
            "For specific questions, use the provided PDF context. "
            "If the context is not sufficient, use your general knowledge of medicine. "
            "If the question is not related to healthcare, politely state that you can only discuss healthcare topics and offer to help with a relevant question. "
            "Be concise and professional. Do not invent information. Always advise consulting a professional medical practitioner for personalized advice."
        )
        user_prompt = f"Question: {question}\n\n[Optional Context from PDF]:\n{relevant_context}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200
        )
        ai_answer = response.choices[0].message.content.strip()
        return JSONResponse({"answer": ai_answer})
    except Exception as e:
        print(f"[ERROR] Chat failed: {e}")
        return JSONResponse({"answer": "Sorry, I couldn't process your request right now."})