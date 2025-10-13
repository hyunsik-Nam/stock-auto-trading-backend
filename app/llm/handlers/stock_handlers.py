import asyncio
import re
from typing import Dict, Any
from .base_handler import BaseHandler
from ..utils.advisor_types import AdvisorState
from ...services.finanace import MarketDataManager
from ..utils.promptManager import YAMLPromptManager
from langchain_core.runnables import RunnableLambda
from app.core.logging_config import getLogger

logger = getLogger(__name__)
class StockOrderHandler(BaseHandler):
    """주식 주문 처리 Handler"""
    
    def __init__(self, model, structured_llm):
        self.model = model
        self.structured_llm = structured_llm
        self.yaml_prompt_manager = YAMLPromptManager()
        self.market_manager = MarketDataManager()
    
    def can_handle(self, classification: Dict[str, Any]) -> bool:
        return classification.get("type", "").upper() == "STOCK_ORDER"
    
    async def handle(self, state: AdvisorState) -> AdvisorState:
        try:
            question = state["question"]
            classification = state.get("stock_classification", {})
            
            def create_order_prompt(data):
                stock_info = f"주식: {data.get('stock', '')}, 액션: {data.get('action', '')}, 수량: {data.get('cnt', 0)}"
                return self.yaml_prompt_manager.create_chat_prompt(
                    "stock_advisor",
                    context=f"주식 주문 정보: {stock_info}",
                    question=f"{question} - 위 정보를 바탕으로 주문을 처리해주세요."
                )
            
            structured_result = (RunnableLambda(create_order_prompt) | self.structured_llm).invoke(classification)
            result = await self._execute_order(structured_result)
            formatted_result = self._format_response(result, "order_confirmation", "transaction")
            
            return {
                **state, 
                "final_result": formatted_result,
                "handler_name": self.handler_name
            }
        except Exception as e:
            return {**state, "error": f"{self.handler_name}: {str(e)}"}
    
    async def _execute_order(self, structured_result) -> Dict[str, Any]:
        """주문 실행 로직"""
        try:
            data = structured_result.get('content', {})
            stock_name = data.get('stock', '')
            action = data.get('action', '')
            quantity = data.get('cnt', 0)
            
            symbol = await self.market_manager.search_korean_stock_symbol(stock_name)
            stock_data = await self.market_manager.get_stock_data(symbol)
            
            return {
                "message": f"{stock_name} {quantity}주 {action} 주문이 완료되었습니다.",
                "stock_name": stock_name,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "stock_data": stock_data.tail(5).to_dict() if not stock_data.empty else {},
                "structured_result": structured_result
            }
        except Exception as e:
            return {"error": f"주문 처리 실패: {str(e)}"}
    
    @property
    def handler_name(self) -> str:
        return "stock_order"

class StockPriceHandler(BaseHandler):
    """주가 조회 Handler"""
    
    def __init__(self, model):
        self.model = model
        self.market_manager = MarketDataManager()
        self.yaml_prompt_manager = YAMLPromptManager()
    
    def can_handle(self, classification: Dict[str, Any]) -> bool:
        return classification.get("type", "").upper() == "STOCK_PRICE"
    
    async def handle(self, state: AdvisorState) -> AdvisorState:
        try:
            question = state["question"]
            stock_name = self._extract_stock_name(question)
            
            print(f"Extracted stock name: {stock_name}")
            
            stock_data = await self.market_manager.search_korean_stock_symbol(stock_name)
            
            logger.info(f"@@@@: {stock_data}")

            if stock_data and len(stock_data) > 0:
                stock_info = stock_data.get('data')[0]
                logger.info(f"Fetched stock data: {stock_info}")
                current_price_str = stock_info.get('현재가', '0').replace('+', '').replace('-', '').replace(',', '')
                current_price = int(current_price_str) if current_price_str.isdigit() else 0
                
                content = {
                    "message": f"{stock_info.get('종목명', stock_name)}의 현재가는 {current_price:,}원입니다.",
                    "stock_name": stock_info.get('종목명', stock_name),
                    "current_price": current_price,
                    "previous_close": stock_info.get('기준가', ''),
                    "change": stock_info.get('전일대비', ''),
                    "volume": stock_info.get('거래량', ''),
                    "stock_data": stock_info
                }
            else:
                logger.info(f"No stock data found for: {stock_name}")
                content = {
                    "message": f"{stock_name}의 주가 정보를 찾을 수 없습니다.",
                    "stock_name": stock_name,
                    "current_price": 0,
                    "stock_data": {}
                }
            
            logger.info(f"Content to be formatted: {content}")
            formatted_result = self._format_response(content, "stock_price", "inquiry")
            
            return {
                **state, 
                "final_result": formatted_result,
                "handler_name": self.handler_name
            }
        except Exception as e:
            return {**state, "error": f"{self.handler_name}: {str(e)}"}
    
    def _extract_stock_name(self, question: str) -> str:
        """질문에서 주식명 추출"""
        patterns = [
            r'([가-힣A-Za-z0-9]+)\s*(?:주가|가격|시세|주식)',
            r'([가-힣A-Za-z0-9]+)\s*(?:얼마|몇|현재)',
            r'([가-힣A-Za-z0-9]+)(?:\s+|의)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                return match.group(1).strip()
        
        return ""  # 기본값
    
    @property
    def handler_name(self) -> str:
        return "stock_price"

class StockAnalysisHandler(BaseHandler):
    """주식 분석 Handler"""
    
    def __init__(self, model, json_parser):
        self.model = model
        self.json_parser = json_parser
        self.market_manager = MarketDataManager()
        self.yaml_prompt_manager = YAMLPromptManager()
    
    def can_handle(self, classification: Dict[str, Any]) -> bool:
        return classification.get("type", "").upper() == "STOCK_ANALYSIS"
    
    async def handle(self, state: AdvisorState) -> AdvisorState:
        try:
            question = state["question"]
            stock_name = self._extract_stock_name(question)
            
            symbol = await self.market_manager.search_korean_stock_symbol(stock_name)
            stock_data = await self.market_manager.get_stock_data(symbol, period="6mo")
            
            # AI 분석 수행
            analysis_context = f"주식 데이터 (최근 10일): {stock_data.tail(10).to_string()}"
            analysis_prompt = self.yaml_prompt_manager.create_chat_prompt(
                "stock_analysis",
                context=analysis_context,
                question=question
            )
            
            analysis_result = (analysis_prompt | self.model | self.json_parser).invoke({"question": question})
            
            content = {
                "analysis": analysis_result,
                "stock_name": stock_name,
                "symbol": symbol,
                "analysis_period": "6개월",
                "stock_data": stock_data.tail(20).to_dict() if not stock_data.empty else {}
            }
            
            formatted_result = self._format_response(content, "stock_analysis", "analysis")
            
            return {
                **state, 
                "final_result": formatted_result,
                "handler_name": self.handler_name
            }
        except Exception as e:
            return {**state, "error": f"{self.handler_name}: {str(e)}"}
    
    def _extract_stock_name(self, question: str) -> str:
        patterns = [
            r'([가-힣A-Za-z0-9]+)\s*(?:분석|전망|추천|어떤)',
            r'([가-힣A-Za-z0-9]+)\s*(?:어떻게|어떤지)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                return match.group(1).strip()
        
        return ""
    
    @property
    def handler_name(self) -> str:
        return "stock_analysis"

class GeneralStockHandler(BaseHandler):
    """일반 주식 상담 Handler"""
    
    def __init__(self, model, json_parser):
        self.model = model
        self.json_parser = json_parser
        self.yaml_prompt_manager = YAMLPromptManager()
    
    def can_handle(self, classification: Dict[str, Any]) -> bool:
        return classification.get("type", "").upper() == "STOCK_GENERAL"
    
    async def handle(self, state: AdvisorState) -> AdvisorState:
        try:
            question = state["question"]
            prompt = self.yaml_prompt_manager.create_chat_prompt(
                "stock_advisor", 
                context="일반적인 주식 상담", 
                question=question
            )
            
            result = (prompt | self.model | self.json_parser).invoke({"question": question})
            formatted_result = self._format_response(result, "stock_advice", "investment")
            
            return {
                **state, 
                "final_result": formatted_result,
                "handler_name": self.handler_name
            }
        except Exception as e:
            return {**state, "error": f"{self.handler_name}: {str(e)}"}
    
    @property
    def handler_name(self) -> str:
        return "stock_general"