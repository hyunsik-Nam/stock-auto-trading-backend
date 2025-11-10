    
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from ..utils.promptManager import YAMLPromptManager
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain.output_parsers.json import SimpleJsonOutputParser
from ..utils.structured_outputs import StockStruct,FinalStockStruct,OrderClassifier
from ..utils.llm_tools import *
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
import json


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

callbacks = [LoggingHandler()]
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
json_parser = SimpleJsonOutputParser()
structured_llm = model.with_structured_output(FinalStockStruct)
yaml_prompt_manager = YAMLPromptManager()



class LLMService:
    def __init__(self):
        pass
    
    def _create_routing_chain(self):
        """라우팅 체인 생성"""
        
        # 각 어드바이저 프롬프트
        def stock_prompt(question: str):
            context = 'test입니다'
            prompt = yaml_prompt_manager.create_chat_prompt("stock_advisor", context=context, question=question)
            return prompt


        # general_prompt = yaml_prompt_manager.create_chat_prompt("general_advisor")

        def general_prompt(question: str) :
            context = 'test입니다'
            prompt = yaml_prompt_manager.create_chat_prompt("general_advisor", context=context, question=question)
            return prompt

        def extract_content(chunk):
            print("extract_content:", chunk)
            return chunk
        
        # 분류기
        classifier =  yaml_prompt_manager.create_chat_prompt("stock_general_branch_prompt") | model
        stock_classifier = yaml_prompt_manager.create_chat_prompt("stock_order_branch") | model.with_structured_output(OrderClassifier)
        

        # 🎯 RunnableBranch로 분기처리
        # routing_chain = RunnableBranch(
        #     (
        #         # 1차 분기: STOCK 여부
        #         lambda x: "STOCK" in classifier.invoke({"question": x["question"]}).content.upper(),
        #         lambda x: RunnableBranch(
        #             (
        #                 # 2차 분기: STOCK_ORDER 여부
        #                 lambda x: "STOCK_ORDER" == stock_classifier.invoke({"question": x["question"]}).content.get("type", ""),
        #                 lambda x: RunnableLambda(extract_content) | parse_stock_info | structured_llm | order_stock
        #             ),
        #             (
        #                 # STOCK_GENERAL일 경우
        #                 lambda x: "STOCK_GENERAL" == stock_classifier.invoke({"question": x["question"]}).content.get("type", ""),
        #                 lambda x: stock_prompt(x["question"]) | model | json_parser
        #             ),
        #         )
        #     ),
        #     (
        #         # 기본값(GENERAL)
        #         lambda x: True,
        #         lambda x: general_prompt(x["question"]) | model | json_parser
        #     )
        # )

        def wrap_stock_data(data):
            return {"stock_data": data}

        # 1차 분기 정의
        routing_chain = RunnableBranch(
            (
                # STOCK 여부 체크
                lambda x: "STOCK" in classifier.invoke({"question": x["question"]}).content.upper(),
                RunnableBranch(
                    (
                        # STOCK_ORDER 체크
                        lambda x: "STOCK_ORDER" == stock_classifier.invoke({"question": x["question"]}).get("type").upper(),
                        parse_stock_info | structured_llm | order_stock
                    ),
                    # 기본값 (다른 STOCK 관련)
                    lambda x: stock_prompt(x["question"]) | model | json_parser
                )
            ),
            lambda x: general_prompt(x["question"]) | model | json_parser
        )
        
        return routing_chain
    

    async def advisor_stream(self, question):
        """상담 스트림"""
        try:
            # 먼저 분류 메시지 전송
            classification_data = {"content": f"상담을 시작합니다.\n\n"}
            yield f"data: {json.dumps(classification_data, ensure_ascii=False)}\n\n"

            # response = await self._create_routing_chain().invoke({"question": question}, config={"callbacks": callbacks})

            # Chain으로 스트리밍
            async for chunk in self._create_routing_chain().astream({"question": question}, config={"callbacks": callbacks}):
                # content가 있으면 한 번에 처리
                content = chunk.get("content", "") if isinstance(chunk, dict) else ""
                # for char in content:
                #     yield f"data: {json.dumps({'content': char}, ensure_ascii=False)}\n\n"
                    # content가 dict면 json 문자열로 변환
                if isinstance(content, dict):
                    content_str = json.dumps(content, ensure_ascii=False)
                    yield f"data: {json.dumps({'content': content_str}, ensure_ascii=False)}\n\n"
                else:
                    for char in content:
                        yield f"data: {json.dumps({'content': char}, ensure_ascii=False)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_data = {"content": f"오류가 발생했습니다: {str(e)}"}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    
