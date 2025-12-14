from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime

class SafetyLabel(str, Enum):
    SAFE = "safe"
    BORDERLINE = "borderline"
    UNSAFE = "unsafe"

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: Role
    content: str

class AttackAttempt(BaseModel):
    id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    strategy_name: str
    prompt_template: str
    input_prompt: str
    target_model: str
    target_response: Optional[str] = None
    safety_score: Optional[float] = None  # 0.0 to 1.0 (1.0 being safest)
    safety_label: Optional[SafetyLabel] = None
    judge_reasoning: Optional[str] = None
    iteration: int = 0
