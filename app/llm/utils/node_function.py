import os
from typing import Any, Dict, List
import traceback

from langchain_google_genai import ChatGoogleGenerativeAI
from .advisor_types import AdvisorState

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.runnables import RunnableLambda
import asyncio
from pathlib import Path

from ..utils.promptManager import YAMLPromptManager
from ..utils.structured_outputs import FinalStockStruct, OrderClassifier
from ..handlers.handler_registry import handler_registry, initialize_handlers
from app.core.logging_config import getLogger

logger = getLogger(__name__)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
logger.info(f"🔍 GOOGLE_API_KEY 로드 상태: {'✅ 설정됨' if GOOGLE_API_KEY else '❌ 없음'}")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 전역 변수 초기화
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    convert_system_message_to_human=True
)
json_parser = SimpleJsonOutputParser()
structured_llm = model.with_structured_output(FinalStockStruct)
yaml_prompt_manager = YAMLPromptManager()

def create_main_classifier():
    """메인 분류기 생성"""
    def _classifyMain(inputs: Dict[str, Any]) -> Any:
        prompt = yaml_prompt_manager.create_chat_prompt("stock_general_branch_prompt")
        return (prompt | model).invoke(inputs)
    
    return RunnableLambda(_classifyMain, name="main_classifier")

def create_stock_classifier():
    """주식 분류기 생성"""
    def _classifyStock(inputs: Dict[str, Any]) -> Any:
        prompt = yaml_prompt_manager.create_chat_prompt("stock_order_branch")
        return (prompt | model.with_structured_output(OrderClassifier)).invoke(inputs)
    
    return RunnableLambda(_classifyStock, name="stock_classifier")

classifier = create_main_classifier()
stock_classifier = create_stock_classifier()

# Handler 초기화
initialize_handlers(model, structured_llm, json_parser)

def classify_main(state: AdvisorState) -> AdvisorState:
    """1차 분류: STOCK vs GENERAL"""
    try:
        question = state["question"]
        logger.info(f"🔍 Main classification for question: {question}")
        
        main_result = classifier.invoke({"question": question})
        
        is_stock = "STOCK" in main_result.content.upper()
        route = "STOCK" if is_stock else "GENERAL"
        
        logger.info(f"✅ Main classification result: {route}")
        
        return {
            **state,
            "main_classification": {"content": main_result.content, "is_stock": is_stock},
            "route": route
        }
    except Exception as e:
        logger.error(f"❌ Main classification error: {e}\n{traceback.format_exc()}")
        return {**state, "error": str(e), "route": "ERROR"}

def classify_stock(state: AdvisorState) -> AdvisorState:
    """2차 분류: 세부 주식 기능 분류"""
    try:
        question = state["question"]
        stock_result = stock_classifier.invoke({"question": question})
        
        # 🔍 타입 검증 및 변환
        logger.info(f"🔍 Stock classification RAW result: {stock_result}")
        logger.info(f"🔍 Stock classification TYPE: {type(stock_result)}")
        
        # Pydantic 모델 → dict 변환
        if not isinstance(stock_result, dict):
            if hasattr(stock_result, 'model_dump'):
                stock_result = stock_result.model_dump()
            elif hasattr(stock_result, 'dict'):
                stock_result = stock_result.dict()
            else:
                logger.warning(f"⚠️ Unexpected type: {type(stock_result)}, converting to dict")
                stock_result = {"type": "STOCK_GENERAL", "stock": "", "action": "", "qty": ""}
        
        # 🎯 type 필드 정규화 (한글 → 영문 변환)
        type_mapping = {
            "주식주문": "STOCK_ORDER",
            "주식일반": "STOCK_GENERAL", 
            "주가조회": "STOCK_PRICE",
            "주식분석": "STOCK_ANALYSIS",
            "주문": "STOCK_ORDER",
            "조회": "STOCK_PRICE",
            "분석": "STOCK_ANALYSIS"
        }
        
        original_type = stock_result.get("type", "")
        normalized_type = type_mapping.get(original_type, original_type)
        
        # 유효한 type 값만 허용
        valid_types = ["STOCK_ORDER", "STOCK_GENERAL", "STOCK_PRICE", "STOCK_ANALYSIS"]
        if normalized_type.upper() not in valid_types:
            logger.warning(f"⚠️ Invalid type '{original_type}', defaulting to STOCK_GENERAL")
            normalized_type = "STOCK_GENERAL"
        
        stock_result["type"] = normalized_type.upper()
        
        logger.info(f"✅ Stock classification FINAL result: {stock_result}")
        
        return {
            **state,
            "stock_classification": stock_result,
            "route": "STOCK_HANDLER"
        }
    except Exception as e:
        logger.error(f"❌ Stock classification error: {e}\n{traceback.format_exc()}")
        return {**state, "error": str(e), "route": "ERROR"}

async def process_stock_with_handlers(state: AdvisorState) -> AdvisorState:
    """Handler 패턴을 사용하는 동적 주식 처리 노드"""
    try:
        classification = state.get("stock_classification", {})
        logger.info(f"🔍 Processing with classification: {classification}")
        
        # Handler 선택
        handler = handler_registry.get_handler(classification)
        
        if handler:
            logger.info(f"🎯 선택된 Handler: {handler.handler_name}")
            result = await handler.handle(state)
            logger.info(f"✅ Handler processing completed: {handler.handler_name}")
            return result
        else:
            error_msg = f"적절한 Handler를 찾을 수 없습니다. Classification: {classification}"
            logger.error(f"❌ {error_msg}")
            return {**state, "error": error_msg, "route": "ERROR"}
            
    except Exception as e:
        logger.error(f"❌ Handler processing error: {e}\n{traceback.format_exc()}")
        return {**state, "error": str(e), "route": "ERROR"}

async def process_general(state: AdvisorState) -> AdvisorState:
    """일반 상담 처리"""
    try:
        logger.info(f"🔍 Processing general question: {state.get('question', '')}")
        
        handler = handler_registry.get_handler_by_name("general_advice")
        
        if handler:
            logger.info(f"🎯 선택된 Handler: {handler.handler_name}")
            result = await handler.handle(state)
            logger.info(f"✅ General processing completed")
            return result
        else:
            error_msg = "General advice handler를 찾을 수 없습니다"
            logger.error(f"❌ {error_msg}")
            return {**state, "error": error_msg, "route": "ERROR"}
            
    except Exception as e:
        logger.error(f"❌ General processing error: {e}\n{traceback.format_exc()}")
        return {**state, "error": str(e), "route": "ERROR"}

def handle_error(state: AdvisorState) -> AdvisorState:
    """에러 처리"""
    error_message = state.get('error', '알 수 없는 오류')
    logger.error(f"🔴 Error handler invoked: {error_message}")
    
    error_result = {
        "content": f"죄송합니다. 처리 중 오류가 발생했습니다: {error_message}",
        "type": "error",
        "category": "system_error",
        "handler": "error_handler",
        "original_question": state.get("question", ""),
        "error_details": error_message
    }
    
    return {**state, "final_result": error_result}

# 노드 함수 메타데이터 설정
classify_main_runnable = RunnableLambda(classify_main, name="classify_main_node")
classify_stock_runnable = RunnableLambda(classify_stock, name="classify_stock_node")
process_stock_with_handlers_runnable = RunnableLambda(process_stock_with_handlers, name="process_stock_with_handlers_node")
process_general_runnable = RunnableLambda(process_general, name="process_general_node")
handle_error_runnable = RunnableLambda(handle_error, name="handle_error_node")