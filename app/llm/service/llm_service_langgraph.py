import json
import asyncio
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END, START
from langchain_core.callbacks import BaseCallbackHandler

from ..utils.advisor_types import AdvisorState
from ..utils.route_function import *
from ..utils.node_function import *
from ..handlers.handler_registry import handler_registry
from app.core.logging_config import getLogger

logger = getLogger(__name__)
class LangGraphCallbackHandler(BaseCallbackHandler):
    """LangGraph 전용 콜백 Handler"""
    
    def __init__(self, stream_callback=None):
        self.stream_callback = stream_callback
    
    def on_chain_start(self, serialized: Optional[Dict[str, Any]], inputs: Dict[str, Any], **kwargs) -> None:
        """체인 시작 시 콜백 - None 안전성 확보"""
        try:
            metadata = kwargs.get('metadata', {})
            # serialized가 None인 경우 처리
            if kwargs.get('name') is None:
                node_name = "unknown_chain"
            else:
                node_name = kwargs.get('name','unknown') if isinstance(kwargs, dict) else "unknown"
            
            logger.info(f"🚀 체인 '{node_name}' 시작")
            logger.info(f"tags: {kwargs.get('tags', [])}")
            logger.info(f"step : {metadata.get('langgraph_step')}")
            logger.info(f"nodename : {metadata.get('langgraph_node')}")
            
            if self.stream_callback:
                self.stream_callback(f"⏳ {node_name} 처리 중...")
                
        except Exception as e:
            print(f"❌ on_chain_start 콜백 오류: {e}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """체인 종료 시 콜백"""
        try:
            print("✅ 노드 처리 완료")
        except Exception as e:
            print(f"❌ on_chain_end 콜백 오류: {e}")
    
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """체인 오류 시 콜백"""
        try:
            print(f"❌ 노드 오류: {error}")
            
            if self.stream_callback:
                self.stream_callback(f"❌ 오류 발생: {str(error)}")
        except Exception as e:
            print(f"❌ on_chain_error 콜백 오류: {e}")
    
    def on_llm_start(self, serialized: Optional[Dict[str, Any]], prompts: List[str], **kwargs) -> None:
        """LLM 시작 시 콜백"""
        try:
            print("🤖 LLM 모델 호출 시작")
            
            if self.stream_callback:
                self.stream_callback("🤖 AI 모델 분석 중...")
        except Exception as e:
            print(f"❌ on_llm_start 콜백 오류: {e}")
    
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """LLM 종료 시 콜백"""
        try:
            print("🎯 LLM 모델 응답 완료")
        except Exception as e:
            print(f"❌ on_llm_end 콜백 오류: {e}")

class LLMServiceGraph:
    """LangGraph + Handler 통합 서비스"""
    
    def __init__(self):
        self.graph = None
        self._callbacks = []
    
    def _setup_callbacks(self, stream_callback=None):
        """콜백 Handler 설정"""
        self._callbacks = [LangGraphCallbackHandler(stream_callback)]
        return self._callbacks
    
    def _create_langgraph_chain(self, callbacks=None):
        """LangGraph 체인 생성 - Handler 통합"""
        
        # 그래프 생성
        workflow = StateGraph(AdvisorState)

        # 노드 추가
        workflow.add_node("classify_main", classify_main_runnable)
        workflow.add_node("classify_stock", classify_stock_runnable)
        workflow.add_node("process_stock_with_handlers", process_stock_with_handlers_runnable)
        workflow.add_node("process_general", process_general_runnable)
        workflow.add_node("handle_error", handle_error_runnable)

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

        # 주식 분류 후 Handler 노드로 라우팅
        workflow.add_conditional_edges(
            "classify_stock",
            route_after_stock_classification,
            {
                "process_stock_with_handlers": "process_stock_with_handlers",
                "handle_error": "handle_error"
            }
        )

        # 모든 처리 노드에서 END로
        workflow.add_edge("process_stock_with_handlers", END)
        workflow.add_edge("process_general", END)
        workflow.add_edge("handle_error", END)

        # 콜백은 compile 시가 아닌 invoke/stream 시에 설정
        return workflow.compile()

    async def advisor_stream(self, question: str):
        """상담 스트림 (LangGraph + Handler 통합)"""
        try:
            # 스트리밍 콜백
            stream_messages = []
            
            def stream_callback(message: str):
                stream_messages.append(message)
            
            # 콜백 설정
            callbacks = self._setup_callbacks(stream_callback)
            
            # 초기 메시지
            classification_data = {"content": "🚀 AI 상담사가 질문을 분석합니다...\n\n"}
            yield f"data: {json.dumps(classification_data, ensure_ascii=False)}\n\n"

            # LangGraph 생성
            graph = self._create_langgraph_chain()
            
            # 🎯 수정된 실행 방식 - RunnableConfig 사용
            from langchain_core.runnables import RunnableConfig
            
            run_config = RunnableConfig(
                callbacks=callbacks,
                tags=["advisor_session","langgraph"],
                metadata={
                    "session_id": "12345", 
                    "handlers_available": len(handler_registry.list_handlers())
                }
            )
            
            # LangGraph 스트리밍 실행
            async for chunk in graph.astream(
                {"question": question},
                config=run_config  # RunnableConfig 객체 전달
            ):
                # 각 노드의 출력 처리
                for node_name, node_output in chunk.items():
                    print(f"📊 Node '{node_name}' output: {type(node_output)}")

                    # 노드별 피드백 메시지
                    node_feedback = {
                        "classify_main": "🔍 질문 유형을 분석하고 있습니다...",
                        "classify_stock": "📈 주식 관련 세부 분류 중...",
                        "process_stock_with_handlers": "💼 전문 Handler가 요청을 처리하고 있습니다...",
                        "process_general": "💭 일반 상담을 처리하고 있습니다...",
                        "handle_error": "🔧 문제를 해결하고 있습니다..."
                    }
                    
                    if node_name in node_feedback:
                        print("@@@@@@@@1")
                        feedback_data = {"content": f"{node_feedback[node_name]}\n\n"}
                        yield f"data: {json.dumps(feedback_data, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.1)

                    # Handler 정보 표시
                    if node_name == "process_stock_with_handlers" and node_output.get("handler_name"):
                        print("@@@@@@@@2")
                        handler_info = {"content": f"🎯 {node_output['handler_name']} Handler가 처리합니다...\n\n"}
                        yield f"data: {json.dumps(handler_info, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.1)

                    # 최종 결과 처리
                    if node_name in ["process_stock_with_handlers", "process_general", "handle_error"]:
                        final_result = node_output.get("final_result")
                        print(f"@@@@@@@2 node_output: {node_output}")
                        print(f"@@@@@@@3 final_result: {final_result}")
                        
                        if final_result:
                            # 타입별 이모지
                            type_emojis = {
                                "stock_advice": "📊 ",
                                "general_advice": "💡 ",
                                "order_confirmation": "✅ ",
                                "stock_price": "💰 ",
                                "stock_analysis": "🔍 ",
                                "error": "❌ "
                            }
                            
                            result_type = final_result.get("type", "")
                            prefix = type_emojis.get(result_type, "📝 ")
                            handler_name = final_result.get("handler", "")
                            
                            # Handler 정보와 함께 프리픽스 표시
                            header = f"{prefix}[{handler_name}] "
                            for char in header:
                                yield f"data: {json.dumps({'content': char}, ensure_ascii=False)}\n\n"
                                await asyncio.sleep(0.02)

                            # 콘텐츠 스트리밍
                            content = final_result.get("content", "")
                            
                            if isinstance(content, dict):
                                # dict인 경우 message 필드 우선 표시
                                if "message" in content:
                                    message = content["message"]
                                    for char in str(message):
                                        yield f"data: {json.dumps({'content': char}, ensure_ascii=False)}\n\n"
                                        await asyncio.sleep(0.03)
                                else:
                                    content_str = json.dumps(content, ensure_ascii=False, indent=2)
                                    for char in content_str:
                                        yield f"data: {json.dumps({'content': char}, ensure_ascii=False)}\n\n"
                                        await asyncio.sleep(0.02)
                            else:
                                for char in str(content):
                                    yield f"data: {json.dumps({'content': char}, ensure_ascii=False)}\n\n"
                                    await asyncio.sleep(0.03)
            
            # 완료 메시지
            completion_data = {"content": "\n\n🎉 상담이 완료되었습니다!"}
            yield f"data: {json.dumps(completion_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"❌ 스트리밍 오류: {e}")
            error_data = {"content": f"❌ 시스템 오류가 발생했습니다: {str(e)}"}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    async def get_graph_info(self):
        """그래프 정보 조회"""
        if not self.graph:
            self.graph = self._create_langgraph_chain()
            
        return {
            "nodes": ["classify_main", "classify_stock", "process_stock_with_handlers", "process_general", "handle_error"],
            "handlers": handler_registry.list_handlers(),
            "total_handlers": len(handler_registry.list_handlers()),
            "callbacks_registered": len(self._callbacks) > 0
        }
    
    async def test_handlers(self):
        """Handler 테스트"""
        test_classifications = [
            {"type": "STOCK_ORDER", "stock": "삼성전자", "action": "매수", "cnt": 10},
            {"type": "STOCK_PRICE", "stock": "LG전자"},
            {"type": "STOCK_ANALYSIS", "stock": "카카오"},
            {"type": "STOCK_GENERAL"},
            {"type": "GENERAL"}
        ]
        
        results = {}
        for classification in test_classifications:
            handler = handler_registry.get_handler(classification)
            results[classification.get("type", "unknown")] = handler.handler_name if handler else "No handler"
        
        return results