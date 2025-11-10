import os
from typing import Any, Dict, List, TypedDict, Literal
import json

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# Your existing imports
from ..utils.promptManager import YAMLPromptManager
from ..utils.structured_outputs import StockStruct, FinalStockStruct, OrderClassifier
from ..utils.llm_tools import *


load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = google_api_key or ""

class LoggingHandler(BaseCallbackHandler):
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat model started")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"Chat model ended, response: {response}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        print(f"Chain {serialized.get('name')} started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain ended, outputs: {outputs}")


# State 정의 - 더 구조화된 상태
class AdvisorState(TypedDict):
    question: str
    main_classification: dict
    stock_classification: dict
    route: str
    processed_data: dict  # 중간 처리 데이터
    final_result: Any
    error: str
    metadata: dict  # 추가 메타데이터


callbacks = [LoggingHandler()]
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
json_parser = SimpleJsonOutputParser()
str_parser = StrOutputParser()
structured_llm = model.with_structured_output(FinalStockStruct)
yaml_prompt_manager = YAMLPromptManager()


class HybridLLMService:
    def __init__(self):
        # LCEL 체인들을 미리 구성
        self._setup_lcel_chains()
    
    def _setup_lcel_chains(self):
        """LCEL 체인들을 미리 구성"""
        
        # 1차 분류 체인
        self.main_classifier_chain = (
            yaml_prompt_manager.create_chat_prompt("stock_general_branch_prompt")
            | model
            | RunnableLambda(self._extract_main_classification)
        )
        
        # 2차 분류 체인 (주식 전용)
        self.stock_classifier_chain = (
            yaml_prompt_manager.create_chat_prompt("stock_order_branch")
            | model.with_structured_output(OrderClassifier)
            | RunnableLambda(self._extract_stock_type)
        )
        
        # 주식 일반 상담 체인
        self.stock_general_chain = (
            RunnableLambda(self._create_stock_prompt)
            | model
            | json_parser
            | RunnableLambda(self._format_stock_response)
        )
        
        # 일반 상담 체인
        self.general_advice_chain = (
            RunnableLambda(self._create_general_prompt)
            | model
            | json_parser
            | RunnableLambda(self._format_general_response)
        )
        
        # 주식 주문 처리 체인
        self.stock_order_chain = (
            RunnableLambda(parse_stock_info)
            | structured_llm
            | order_stock
            | RunnableLambda(self._format_order_response)
        )
    
    # LCEL 체인에서 사용할 헬퍼 함수들
    def _extract_main_classification(self, llm_result):
        """1차 분류 결과 추출"""
        content = llm_result.content if hasattr(llm_result, 'content') else str(llm_result)
        is_stock = "STOCK" in content.upper()
        return {
            "content": content,
            "is_stock": is_stock,
            "confidence": self._calculate_confidence(content)
        }
    
    def _extract_stock_type(self, structured_result):
        """2차 분류 결과 추출"""
        if hasattr(structured_result, 'dict'):
            return structured_result.dict()
        elif isinstance(structured_result, dict):
            return structured_result
        else:
            return {"type": "STOCK_GENERAL", "confidence": 0.5}
    
    def _create_stock_prompt(self, data):
        """주식 프롬프트 생성"""
        question = data.get("question", "")
        context = data.get("context", "test입니다")
        return yaml_prompt_manager.create_chat_prompt(
            "stock_advisor", 
            context=context, 
            question=question
        )
    
    def _create_general_prompt(self, data):
        """일반 프롬프트 생성"""
        question = data.get("question", "")
        context = data.get("context", "test입니다")
        return yaml_prompt_manager.create_chat_prompt(
            "general_advisor",
            context=context,
            question=question
        )
    
    def _format_stock_response(self, response):
        """주식 응답 포맷팅"""
        return {
            "content": response,
            "type": "stock_advice",
            "category": "investment"
        }
    
    def _format_general_response(self, response):
        """일반 응답 포맷팅"""
        return {
            "content": response,
            "type": "general_advice",
            "category": "general"
        }
    
    def _format_order_response(self, response):
        """주문 응답 포맷팅"""
        return {
            "content": response,
            "type": "order_confirmation",
            "category": "transaction"
        }
    
    def _calculate_confidence(self, content):
        """신뢰도 계산 (간단한 키워드 기반)"""
        stock_keywords = ["주식", "투자", "매수", "매도", "종목", "주가"]
        keyword_count = sum(1 for keyword in stock_keywords if keyword in content)
        return min(keyword_count * 0.2 + 0.1, 1.0)
    
    def _create_langgraph_chain(self):
        """LangGraph + LCEL 하이브리드 체인 생성"""
        
        # 노드 함수들 - LCEL 체인을 활용
        def classify_main_node(state: AdvisorState) -> AdvisorState:
            """1차 분류 노드 - LCEL 체인 사용"""
            try:
                # LCEL 체인으로 분류 수행
                main_result = self.main_classifier_chain.invoke({
                    "question": state["question"]
                })
                
                route = "STOCK" if main_result["is_stock"] else "GENERAL"
                
                return {
                    **state,
                    "main_classification": main_result,
                    "route": route,
                    "metadata": {"step": "main_classification", "confidence": main_result.get("confidence", 0)}
                }
            except Exception as e:
                return {**state, "error": str(e), "route": "ERROR"}

        def classify_stock_node(state: AdvisorState) -> AdvisorState:
            """2차 분류 노드 - LCEL 체인 사용"""
            try:
                # LCEL 체인으로 주식 분류 수행
                stock_result = self.stock_classifier_chain.invoke({
                    "question": state["question"]
                })
                
                stock_type = stock_result.get("type", "").upper()
                
                return {
                    **state,
                    "stock_classification": stock_result,
                    "route": stock_type,
                    "metadata": {**state.get("metadata", {}), "stock_type": stock_type}
                }
            except Exception as e:
                return {**state, "error": str(e), "route": "ERROR"}

        def process_stock_order_node(state: AdvisorState) -> AdvisorState:
            """주식 주문 처리 노드 - LCEL 체인 사용"""
            try:
                # LCEL 체인으로 주문 처리
                result = self.stock_order_chain.invoke(state["stock_classification"])
                
                return {
                    **state, 
                    "final_result": result,
                    "processed_data": {"order_processed": True}
                }
            except Exception as e:
                return {**state, "error": str(e)}

        def process_stock_general_node(state: AdvisorState) -> AdvisorState:
            """주식 일반 상담 노드 - LCEL 체인 사용"""
            try:
                # LCEL 체인으로 주식 상담 처리
                result = self.stock_general_chain.invoke({
                    "question": state["question"],
                    "context": "주식 전문 상담"
                })
                
                return {
                    **state,
                    "final_result": result,
                    "processed_data": {"advice_type": "stock_general"}
                }
            except Exception as e:
                return {**state, "error": str(e)}

        def process_general_node(state: AdvisorState) -> AdvisorState:
            """일반 상담 노드 - LCEL 체인 사용"""
            try:
                # LCEL 체인으로 일반 상담 처리
                result = self.general_advice_chain.invoke({
                    "question": state["question"],
                    "context": "일반 상담"
                })
                
                return {
                    **state,
                    "final_result": result,
                    "processed_data": {"advice_type": "general"}
                }
            except Exception as e:
                return {**state, "error": str(e)}

        def handle_error_node(state: AdvisorState) -> AdvisorState:
            """에러 처리 노드 - 구조화된 에러 응답"""
            error_chain = (
                RunnableLambda(lambda x: {
                    "error": x.get("error", "알 수 없는 오류"),
                    "step": x.get("metadata", {}).get("step", "unknown"),
                    "question": x.get("question", "")
                })
                | RunnableLambda(lambda x: {
                    "content": f"죄송합니다. {x['step']} 단계에서 오류가 발생했습니다: {x['error']}",
                    "type": "error",
                    "recovery_suggestion": "다시 질문해 주시거나 문의사항을 남겨주세요."
                })
            )
            
            error_result = error_chain.invoke(state)
            return {**state, "final_result": error_result}

        # 라우팅 함수들 - 더 정교한 조건 처리
        def route_after_main_classification(state: AdvisorState) -> Literal["classify_stock", "process_general", "handle_error"]:
            """메인 분류 후 라우팅 - 개선된 조건부 로직"""
            route = state.get("route", "")
            main_classification = state.get("main_classification", {})
            
            # 에러 처리
            if route == "ERROR":
                return "handle_error"
            elif route == "STOCK":
                # 신뢰도 기반 추가 검증
                confidence = main_classification.get("confidence", 0)
                if confidence < 0.3:  # 신뢰도가 낮으면 일반 처리
                    return "process_general"
                return "classify_stock"
            else:
                # 키워드 기반 2차 검증
                question = state.get("question", "").lower()
                stock_keywords = ["삼성", "lg", "현대", "주가", "투자", "매수", "매도"]
                if any(keyword in question for keyword in stock_keywords):
                    return "classify_stock"
                return "process_general"

        def route_after_stock_classification(state: AdvisorState) -> Literal["process_stock_order", "process_stock_general", "handle_error"]:
            """주식 분류 후 라우팅"""
            route = state.get("route", "")
            
            if route == "ERROR":
                return "handle_error"
            elif route == "STOCK_ORDER":
                return "process_stock_order"
            else:
                return "process_stock_general"

        # 그래프 생성
        workflow = StateGraph(AdvisorState)

        # 노드 추가
        workflow.add_node("classify_main", classify_main_node)
        workflow.add_node("classify_stock", classify_stock_node)
        workflow.add_node("process_stock_order", process_stock_order_node)
        workflow.add_node("process_stock_general", process_stock_general_node)
        workflow.add_node("process_general", process_general_node)
        workflow.add_node("handle_error", handle_error_node)

        # 엣지 추가
        workflow.add_edge(START, "classify_main")
        
        # 메인 분류 후 조건부 라우팅
        workflow.add_conditional_edges(
            "classify_main",
            route_after_main_classification,
            {
                "classify_stock": "classify_stock",
                "process_general": "process_general",
                "handle_error": "handle_error"
            }
        )

        # 주식 분류 후 조건부 라우팅
        workflow.add_conditional_edges(
            "classify_stock",
            route_after_stock_classification,
            {
                "process_stock_order": "process_stock_order",
                "process_stock_general": "process_stock_general",
                "handle_error": "handle_error"
            }
        )

        # 모든 처리 노드에서 END로
        workflow.add_edge("process_stock_order", END)
        workflow.add_edge("process_stock_general", END)
        workflow.add_edge("process_general", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    async def advisor_stream(self, question):
        """상담 스트림 (하이브리드 버전)"""
        try:
            # 시작 메시지
            classification_data = {"content": f"🤖 AI 상담을 시작합니다...\n\n"}
            yield f"data: {json.dumps(classification_data, ensure_ascii=False)}\n\n"

            # LangGraph 실행
            graph = self._create_langgraph_chain()
            
            # 스트리밍 실행
            async for chunk in graph.astream(
                {
                    "question": question,
                    "metadata": {"session_id": "stream", "timestamp": json.dumps({}, default=str)},
                    "processed_data": {}
                }, 
                config={"callbacks": callbacks}
            ):
                # 중간 단계 정보도 스트리밍 (선택적)
                for node_name, node_output in chunk.items():
                    # 분류 단계 피드백
                    if node_name == "classify_main":
                        route = node_output.get("route", "")
                        if route == "STOCK":
                            feedback = {"content": "📈 주식 관련 질문으로 분류되었습니다...\n"}
                            yield f"data: {json.dumps(feedback, ensure_ascii=False)}\n\n"
                        elif route == "GENERAL":
                            feedback = {"content": "💬 일반 상담으로 분류되었습니다...\n"}
                            yield f"data: {json.dumps(feedback, ensure_ascii=False)}\n\n"
                    
                    # 최종 결과 스트리밍
                    elif node_name in ["process_stock_order", "process_stock_general", "process_general", "handle_error"]:
                        final_result = node_output.get("final_result")
                        if final_result:
                            content = self._extract_content_for_streaming(final_result)
                            
                            # 타입별 이모지 추가
                            type_emojis = {
                                "stock_advice": "📊 ",
                                "general_advice": "💡 ",
                                "order_confirmation": "✅ ",
                                "error": "❌ "
                            }
                            
                            result_type = final_result.get("type", "")
                            prefix = type_emojis.get(result_type, "")
                            
                            # 문자별 스트리밍
                            full_content = prefix + content
                            for char in full_content:
                                yield f"data: {json.dumps({'content': char}, ensure_ascii=False)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_data = {"content": f"❌ 시스템 오류가 발생했습니다: {str(e)}"}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    
    def _extract_content_for_streaming(self, result):
        """스트리밍용 컨텐츠 추출"""
        if isinstance(result, dict):
            content = result.get("content", "")
            if isinstance(content, dict):
                return json.dumps(content, ensure_ascii=False, indent=2)
            return str(content)
        return str(result)

    # 동기식 처리 메소드 (테스트용)
    def process_question(self, question: str) -> dict:
        """동기식 질문 처리"""
        graph = self._create_langgraph_chain()
        result = graph.invoke({"question": question})
        return result

    # 디버깅용 메소드
    def debug_classification(self, question: str) -> dict:
        """분류 과정 디버깅"""
        main_result = self.main_classifier_chain.invoke({"question": question})
        
        debug_info = {
            "question": question,
            "main_classification": main_result,
        }
        
        if main_result.get("is_stock"):
            stock_result = self.stock_classifier_chain.invoke({"question": question})
            debug_info["stock_classification"] = stock_result
        
        return debug_info