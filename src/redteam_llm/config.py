from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class RedTeamConfig:
    attacker_model: str
    target_model: str
    judge_model: str
    strategies: List[str]
    payloads: List[str]
    iterations: int = 1
    g2pia: "G2PIAConfig" = field(default=None)


@dataclass
class G2PIAConfig:
    epsilon: float = 0.2
    gamma: float = 0.5
    delta: float = 0.05
    N: int = 50
    max_attempts: int = 500
    generator_model: str = "phi3:mini"
    allow_adversarial_sim: bool = False


def load_config(path: str) -> RedTeamConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    g2pia_cfg = data.get("g2pia", {}) if isinstance(data, dict) else {}
    g2pia_settings = (
        G2PIAConfig(
            epsilon=g2pia_cfg.get("epsilon", 0.2),
            gamma=g2pia_cfg.get("gamma", 0.5),
            delta=g2pia_cfg.get("delta", 0.05),
            N=g2pia_cfg.get("N", 50),
            max_attempts=g2pia_cfg.get("max_attempts", 500),
            generator_model=g2pia_cfg.get("generator_model", "phi3:mini"),
            allow_adversarial_sim=g2pia_cfg.get("allow_adversarial_sim", False),
        )
        if isinstance(g2pia_cfg, dict)
        else G2PIAConfig()
    )
    return RedTeamConfig(
        attacker_model=data.get("attacker_model", "llama3"),
        target_model=data.get("target_model", "llama3"),
        judge_model=data.get("judge_model", "llama3"),
        strategies=data.get("strategies", ["Direct Request", "Roleplay"]),
        payloads=data.get("payloads", ["[TEST PAYLOAD]"]),
        iterations=data.get("iterations", 1),
        g2pia=g2pia_settings,
    )
