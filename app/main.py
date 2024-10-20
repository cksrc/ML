
from fastapi import FastAPI
from app.routers import llm

app = FastAPI()

app.include_router(llm.router)


