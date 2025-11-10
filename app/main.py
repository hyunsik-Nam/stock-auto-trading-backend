from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# 모든 라우터 임포트 (Spring Boot의 Controller 스캔과 같음)
from app.routers import chat, users, items, orders, finance
from app.core.logging_config import setupGlobalLogging, getLogger

# 전역 로깅 초기화
setupGlobalLogging(
    logLevel="INFO",
    enableConsole=True,
    enableFile=False,
    enableRotation=False
)

# 메인 로거 생성
logger = getLogger(__name__)

load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="Finance Model API",
    description="Spring Boot 스타일의 FastAPI 프로젝트",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🎯 모든 라우터 등록 (Spring Boot의 자동 컨트롤러 스캔과 같은 효과)
app.include_router(users.router, prefix="/api/v1/users", tags=["사용자 관리"])
app.include_router(items.router, prefix="/api/v1/items", tags=["상품 관리"])  
app.include_router(orders.router, prefix="/api/v1/orders", tags=["주문 관리"])
app.include_router(finance.router, prefix="/api/v1/finance", tags=["재무 관리"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["챗 경로"])


@app.get("/")
def root():
    return {
        "message": "🚀 전체 프로젝트가 실행되고 있습니다!",
        "available_endpoints": {
            "users": "/api/v1/users",
            "items": "/api/v1/items", 
            "orders": "/api/v1/orders",
            "finance": "/api/v1/finance",
            "test": "/api/v1/test"
        },
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "message": "모든 서비스가 정상 작동 중입니다"}

# 직접 실행 지원
if __name__ == "__main__":
    print("🚀 Spring Boot 스타일 FastAPI 서버 시작...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
