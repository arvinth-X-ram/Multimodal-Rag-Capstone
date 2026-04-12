from fastapi import FastAPI
app=FastAPI()

from src.api.v1.routes import admin,query

@app.get("/health-check")
def health_check():
    return{
        "msg":"Project Running"
    }

app.include_router(admin.router,prefix="/api/v1")
app.include_router(query.router,prefix="/api/v1")