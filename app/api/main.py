from fastapi import FastAPI
from pydantic import BaseModel

from app.services.qa_service import QAService

app = FastAPI(title="Internal Chatbot")

qa_service = QAService()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    answer = qa_service.answer(req.question)
    return {"answer": answer}
