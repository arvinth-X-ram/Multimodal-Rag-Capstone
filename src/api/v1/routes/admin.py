from fastapi import APIRouter

router=APIRouter(prefix="/admin",tags=["admin"])

@router.post("/upload")
def admin():
    return {
        "msg":"Admin Upload"
    }