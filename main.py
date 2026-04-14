from fastapi import FastAPI
<<<<<<< HEAD
from src.api.v1.routes import query
app=FastAPI()

@app.get("/")
def read_root():
    return {"Message": "Hello World"}

@app.get("/health")
def health_check():
    return {
        "status":"ok"
    }

app.include_router(query.router,prefix="/api/v1")
=======
app=FastAPI()

from src.api.v1.routes import admin,query

@app.get("/health-check")
def health_check():
    return{
        "msg":"Project Running"
    }

app.include_router(admin.router,prefix="/api/v1")
app.include_router(query.router,prefix="/api/v1")
>>>>>>> raghul
