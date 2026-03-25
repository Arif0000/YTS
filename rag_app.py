import os
import re
from dotenv import load_dotenv
from typing import List, TypedDict

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------
# ENV
# ---------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ---------------------------
# MODELS
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# 🔥 LOCAL embeddings (NO API LIMIT)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ---------------------------
# HELPERS
# ---------------------------

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([^&]+)", url)
    return match.group(1) if match else None


def get_transcript(video_url):
    video_id = extract_video_id(video_url)

    if not video_id:
        return ""

    try:
        data = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = " ".join([t.text for t in data])
        return transcript
    except:
        return ""


def create_vector_store(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = splitter.create_documents([transcript])

    # limit chunks (important)
    docs = docs[:50]

    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store


# ---------------------------
# MAIN FUNCTIONS
# ---------------------------

def chat_with_video(video_url: str, question: str):
    transcript = get_transcript(video_url)

    if not transcript:
        return "No transcript available for this video."

    vector_store = create_vector_store(transcript)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a precise and reliable AI assistant for question answering.

Your task is to answer the user's question using ONLY the provided context.

Instructions:
- Read the context carefully before answering.
- Extract only the information that is directly relevant to the question.
- Do NOT use prior knowledge or make assumptions beyond the context.
- If the answer is not explicitly or clearly present in the context, respond with:
  "I don't know"

Answering Rules:
- Be concise but complete.
- Use clear and direct language.
- If applicable, include key facts, definitions, or steps from the context.
- Do not repeat the entire context—only include necessary information.
- Do not fabricate or infer missing details.

Advanced Constraints:
- If the context contains multiple relevant pieces, synthesize them into a coherent answer.
- If the question is ambiguous and cannot be resolved using context alone, respond:
  "I don't know"
- If numerical or factual data is present, ensure exact accuracy.

Output Format:
- Provide a direct answer only (no explanations about the process).
- Do not include phrases like "Based on the context" or "According to the given text".

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content


def summarize_video(video_url: str):
    transcript = get_transcript(video_url)

    if not transcript:
        return "No transcript available for this video."

    prompt = f"""
You are an expert content summarizer.

Your task is to analyze a YouTube transcript and produce a high-quality summary.

Instructions:
- Carefully read the full transcript before summarizing.
- Extract the most important ideas, concepts, and insights.
- Avoid repetition, filler, and irrelevant details.
- Focus on clarity, accuracy, and meaningful compression of information.

Output Rules:
- Format the summary strictly as bullet points.
- If the transcript contains sufficient information, output EXACTLY 10 bullet points.
- If the transcript is short or lacks depth, output fewer bullet points (but only meaningful ones).
- Each bullet point should be concise but informative (1–2 lines max).
- Maintain logical flow and coverage of the entire content.

Style Guidelines:
- Use simple, clear language.
- Preserve technical terms if present.
- Do not hallucinate or add information not present in the transcript.
- Do not include introductions, conclusions, or explanations—only bullet points.

Transcript:
{transcript}
"""

    response = llm.invoke(prompt)
    return response.content