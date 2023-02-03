from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from app import predict_answer
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request, response: Response, text: str = None):
    if text:
        answer = predict_answer(text).replace("AI: ",'')
    return templates.TemplateResponse("form_and_answer.html", {"request": request, "text": text, "answer": answer})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)