import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path

# 로그 디렉토리 생성
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# 로그 레벨 매핑
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def setupGlobalLogging(
    logLevel: str = "INFO",
    enableConsole: bool = True,
    enableFile: bool = True,
    enableRotation: bool = True
) -> None:
    """전역 로깅 설정"""
    
    # 기본 로거 설정
    rootLogger = logging.getLogger()
    rootLogger.setLevel(LOG_LEVELS.get(logLevel.upper(), logging.INFO))
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in rootLogger.handlers[:]:
        rootLogger.removeHandler(handler)
    
    # 공통 포맷터
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    if enableConsole:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setLevel(logging.INFO)
        consoleHandler.setFormatter(formatter)
        rootLogger.addHandler(consoleHandler)
    
    # 파일 핸들러
    if enableFile:
        if enableRotation:
            # 로테이션 파일 핸들러 (10MB, 5개 백업)
            fileHandler = logging.handlers.RotatingFileHandler(
                LOG_DIR / "app.log",
                maxBytes=10*1024*1024,
                backupCount=5,
                encoding='utf-8'
            )
        else:
            # 일반 파일 핸들러
            fileHandler = logging.FileHandler(
                LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log",
                encoding='utf-8'
            )
        
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(formatter)
        rootLogger.addHandler(fileHandler)
    
    # 에러 전용 파일 핸들러
    errorHandler = logging.FileHandler(
        LOG_DIR / "error.log",
        encoding='utf-8'
    )
    errorHandler.setLevel(logging.ERROR)
    errorHandler.setFormatter(formatter)
    rootLogger.addHandler(errorHandler)

def getLogger(name: str) -> logging.Logger:
    """모듈별 로거 생성"""
    return logging.getLogger(name)

# 모듈별 로거 설정
def configureModuleLogger(moduleName: str, level: str = "INFO") -> logging.Logger:
    """특정 모듈용 로거 설정"""
    logger = logging.getLogger(moduleName)
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    return logger