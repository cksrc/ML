
from fastapi import FastAPI
from app.routers import llm, pd

app = FastAPI()

app.include_router(llm.router)
app.include_router(pd.router)


