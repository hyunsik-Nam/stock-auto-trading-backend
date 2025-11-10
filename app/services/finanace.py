import os
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Union
import aiohttp
import asyncio
import sys
from app.core.logging_config import getLogger

logger = getLogger(__name__)

BASE_URL = os.getenv("KIWOOM_BASE_URL", "http://localhost:8080/api/v1/kiwoom/")

class MarketDataManager:
    """시장 데이터 관리 클래스"""
    
    def __init__(self):
        self._dataCache = {}
        self._baseUrl = BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def _getSession(self) -> aiohttp.ClientSession:
        """HTTP 세션 관리"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=aiohttp.TCPConnector(limit=10)
            )
        return self._session
    
    def getStockData(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """주식 데이터 조회 - 동기"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            self._dataCache[symbol] = data
            return data
        except Exception as e:
            logger.error(f"데이터 조회 오류 ({symbol}): {e}")
            return pd.DataFrame()
    
    def getRealTimePrice(self, symbol: str) -> float:
        """실시간 가격 조회 - 동기"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('currentPrice', 0)
        except Exception as e:
            logger.error(f"가격 조회 오류 ({symbol}): {e}")
            return 0.0

    async def searchKoreanStockSymbol(self, stockName: str) -> Union[List[Dict], str]:
        """한국 주식 심볼 검색 - 완전 비동기"""
        logger.info(f"searchKoreanStockSymbol called with stockName: {stockName}")
        print(f"searchKoreanStockSymbol called with stockName: {stockName}")
        
        try:
            session = await self._getSession()
            url = f"{self._baseUrl}stock_info/{stockName}"
            
            # 🎯 aiohttp로 비동기 HTTP 요청
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()  # 🎯 올바른 비동기 json() 호출
                    logger.info(f"파싱된 데이터: {data}")
                    return data
                else:
                    logger.warning(f"API 호출 실패: {response.status}")
                    return f"{stockName}.KS"
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP 클라이언트 오류 ({stockName}): {e}")
            return f"{stockName}.KS"
        except asyncio.TimeoutError:
            logger.error(f"API 호출 타임아웃 ({stockName})")
            return f"{stockName}.KS"
        except Exception as e:
            logger.error(f"예상치 못한 오류 ({stockName}): {e}")
            return f"{stockName}.KS"
    
    async def closeSession(self):
        """세션 정리"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    # 기존 함수명과의 호환성을 위한 별칭
    def get_stock_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        return self.getStockData(symbol, period)
    
    def get_real_time_price(self, symbol: str) -> float:
        return self.getRealTimePrice(symbol)
    
    async def search_korean_stock_symbol(self, stock_name: str) -> Union[List[Dict], str]:
        return await self.searchKoreanStockSymbol(stock_name)

class RiskManager:
    """리스크 관리 클래스"""
    
    def __init__(self, maxDrawdown: float = 0.15, maxSingleLoss: float = 0.05):
        self._maxDrawdown = maxDrawdown
        self._maxSingleLoss = maxSingleLoss
        self._peakValue = 0
        
    def checkRiskLimits(self, currentValue: float, trades: List[Dict]) -> Dict:
        """리스크 한계 점검"""
        if currentValue > self._peakValue:
            self._peakValue = currentValue
            
        drawdown = (self._peakValue - currentValue) / self._peakValue if self._peakValue > 0 else 0
        
        # 최근 거래의 손실 체크
        recentLoss = 0
        if trades:
            latestTrade = trades[-1]
            if latestTrade.get('pnl', 0) < 0:
                recentLoss = abs(latestTrade['pnl']) / currentValue
        
        riskStatus = {
            "currentDrawdown": drawdown,
            "maxDrawdownExceeded": drawdown > self._maxDrawdown,
            "singleLossExceeded": recentLoss > self._maxSingleLoss,
            "tradingAllowed": drawdown <= self._maxDrawdown and recentLoss <= self._maxSingleLoss
        }
        
        return riskStatus
    
    # 기존 함수명과의 호환성을 위한 별칭
    def check_risk_limits(self, current_value: float, trades: List[Dict]) -> Dict:
        return self.checkRiskLimits(current_value, trades)