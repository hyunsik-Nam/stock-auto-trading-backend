from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal, TypedDict

class OrderClassifier(TypedDict):
    """주식 분류 결과"""
    stock: Annotated[str, ..., "종목명 (예: 삼성전자, 애플)"]
    qty: Annotated[str, ..., "수량 (예: 10)"]
    action: Annotated[str, ..., "매수/매도 (예: BUY, SELL)"]
    type: Annotated[
        Literal["STOCK_ORDER", "STOCK_GENERAL", "STOCK_PRICE", "STOCK_ANALYSIS"],
        ...,
        "분류 타입 - 반드시 STOCK_ORDER, STOCK_GENERAL, STOCK_PRICE, STOCK_ANALYSIS 중 하나"
    ]
class StockStruct(TypedDict):
    """ 주식정보를 담는 데이터 구조 """

    stock: Annotated[str, ..., "주식 설정 정보", "example=삼성전자, 애플"]
    current_price: Annotated[float, ..., "현재 주식 가격", "example=1000.0"]
    target_price: Annotated[float, None, "목표 주식 가격", "example=1200.0"]
    stop_loss: Annotated[Optional[float], None, "손절가", "example=900.0"]
    take_profit: Annotated[Optional[float], None, "익절가", "example=1100.0"]
    cnt: Annotated[int, ..., "주식 수량 (정수만 허용)", "example=10"]
    action: Annotated[str, ..., "주식 주문 액션", "example=BUY, 매수"]

class FinalStockStruct(TypedDict):
    """ 최종 주식정보를 담는 데이터 구조 """
    content: StockStruct

class GeneralStruct(TypedDict):
    """ 일반 정보를 담는 데이터 구조 """
    content: Annotated[str, ..., "질문에 대한 답변", "example=이것은 예시 답변입니다."]