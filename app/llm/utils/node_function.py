import os
from typing import Any, Dict, List
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


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"🔍 GOOGLE_API_KEY 로드 상태: {'✅ 설정됨' if GOOGLE_API_KEY else '❌ 없음'}")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다. .env 파일에 GOOGLE_API_KEY=your_api_key를 추가해주세요.")

# 환경변수에 명시적으로 설정 (langchain이 인식할 수 있도록)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY



# 전역 변수 초기화
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
json_parser = SimpleJsonOutputParser()
structured_llm = model.with_structured_output(FinalStockStruct)
yaml_prompt_manager = YAMLPromptManager()

# 분류기들
# classifier = RunnableLambda(yaml_prompt_manager.create_chat_prompt("stock_general_branch_prompt") | model, name="main_classifier")
# stock_classifier = RunnableLambda(yaml_prompt_manager.create_chat_prompt("stock_order_branch") | model.with_structured_output(OrderClassifier), name ="stock_classifier")

def create_main_classifier():
    """메인 분류기 생성 - 콜백 최적화"""
    def _classifyMain(inputs: Dict[str, Any]) -> Any:
        prompt = yaml_prompt_manager.create_chat_prompt("stock_general_branch_prompt")
        return (prompt | model).invoke(inputs)
    
    return RunnableLambda(_classifyMain, name="main_classifier")

def create_stock_classifier():
    """주식 분류기 생성 - 콜백 최적화"""
    def _classifyStock(inputs: Dict[str, Any]) -> Any:
        prompt = yaml_prompt_manager.create_chat_prompt("stock_order_branch")
        return (prompt | model.with_structured_output(OrderClassifier)).invoke(inputs)
    
    return RunnableLambda(_classifyStock, name="stock_classifier")

# 분류기들 - 단일 콜백만 발생하도록 최적화
classifier = create_main_classifier()
stock_classifier = create_stock_classifier()

# Handler들 초기화
initialize_handlers(model, structured_llm, json_parser)

def classify_main(state: AdvisorState) -> AdvisorState:
    """1차 분류: STOCK vs GENERAL"""
    try:
        question = state["question"]
        print(f"🔍 Main classification for question: {question}")
        main_result = classifier.invoke({"question": question})
        
        is_stock = "STOCK" in main_result.content.upper()
        route = "STOCK" if is_stock else "GENERAL"
        
        return {
            **state,
            "main_classification": {"content": main_result.content, "is_stock": is_stock},
            "route": route
        }
    except Exception as e:
        print(f"❌ Main classification error: {e}")
        return {**state, "error": str(e), "route": "ERROR"}

def classify_stock(state: AdvisorState) -> AdvisorState:
    """2차 분류: 세부 주식 기능 분류"""
    try:
        question = state["question"]
        stock_result = stock_classifier.invoke({"question": question})
        
        return {
            **state,
            "stock_classification": stock_result,
            "route": "STOCK_HANDLER"
        }
    except Exception as e:
        print(f"❌ Stock classification error: {e}")
        return {**state, "error": str(e), "route": "ERROR"}

async def process_stock_with_handlers(state: AdvisorState) -> AdvisorState:
    """Handler 패턴을 사용하는 동적 주식 처리 노드"""
    try:
        classification = state.get("stock_classification", {})
        
        # 적절한 Handler 선택
        handler = handler_registry.get_handler(classification)
        
        if handler:
            print(f"🎯 선택된 Handler: {handler.handler_name}")
            return await handler.handle(state)
        else:
            raise Exception("적절한 Handler를 찾을 수 없습니다")
            
    except Exception as e:
        print(f"❌ Handler processing error: {e}")
        return {**state, "error": str(e)}

async def process_general(state: AdvisorState) -> AdvisorState:
    """일반 상담 처리"""
    try:
        # 일반 상담도 Handler를 통해 처리
        handler = handler_registry.get_handler_by_name("general_advice")
        
        if handler:
            print(f"🎯 선택된 Handler: {handler.handler_name}")
            return await handler.handle(state)
        else:
            raise Exception("General advice handler를 찾을 수 없습니다")
            
    except Exception as e:
        print(f"❌ General processing error: {e}")
        return {**state, "error": str(e)}

def handle_error(state: AdvisorState) -> AdvisorState:
    """에러 처리"""
    error_message = state.get('error', '알 수 없는 오류')
    error_result = {
        "content": f"오류가 발생했습니다: {error_message}",
        "type": "error",
        "category": "system_error",
        "handler": "error_handler"
    }
    return {**state, "final_result": error_result}

    # 노드 함수 메타데이터 설정
classify_main_runnable = RunnableLambda(classify_main, name="classify_main_node")
classify_stock_runnable = RunnableLambda(classify_stock, name="classify_stock_node")
process_stock_with_handlers_runnable = RunnableLambda(process_stock_with_handlers, name="process_stock_with_handlers_node")
process_general_runnable = RunnableLambda(process_general, name="process_general_node")
handle_error_runnable = RunnableLambda(handle_error, name="handle_error_node")