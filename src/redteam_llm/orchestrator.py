from .attacker import Attacker
from .judge import SafetyJudge
from .llm import LLMInterface
from .types import AttackAttempt, SafetyLabel, Message, Role
from .config import RedTeamConfig
import uuid
from typing import List

class RedTeamOrchestrator:
    def __init__(self, config: RedTeamConfig):
        self.config = config
        self.attacker = Attacker(config.attacker_model)
        self.judge = SafetyJudge(config.judge_model)
        self.target = LLMInterface(config.target_model)
        self.results: List[AttackAttempt] = []

    def run_session(self):
        print(f"Starting Red Team Session against {self.config.target_model}")
        
        for payload in self.config.payloads:
            print(f"\nTesting Payload: {payload}")
            for strategy in self.config.strategies:
                print(f"  Strategy: {strategy}")
                self._run_attack_loop(payload, strategy)

    def _run_attack_loop(self, payload: str, strategy: str):
        feedback = None
        
        for i in range(self.config.iterations):
            # 1. Generate Attack Template
            template = self.attacker.generate_attack(strategy, feedback)
            
            # 2. Fill Payload
            # Safety check: Ensure we don't accidentally execute code or bad formatting
            try:
                prompt_input = template.replace("{PAYLOAD}", payload)
            except Exception:
                prompt_input = f"{template} {payload}"

            # 3. Target Response
            target_response = self.target.generate_text(prompt_input)
            
            # 4. Evaluate
            evaluation = self.judge.evaluate(prompt_input, target_response)
            
            attempt = AttackAttempt(
                id=str(uuid.uuid4()),
                strategy_name=strategy,
                prompt_template=template,
                input_prompt=prompt_input,
                target_model=self.config.target_model,
                target_response=target_response,
                safety_score=evaluation.get('safety_score'),
                safety_label=evaluation.get('label'),
                judge_reasoning=evaluation.get('reasoning'),
                iteration=i + 1
            )
            self.results.append(attempt)
            
            print(f"    Iter {i+1}: {attempt.safety_label} (Score: {attempt.safety_score})")
            
            # Stop if unsafe (attack succeeded)
            if attempt.safety_label == SafetyLabel.UNSAFE:
                print("    [!] Attack Succeeded!")
                break
                
            # Update feedback for next iteration
            feedback = evaluation.get('reasoning', "Attack failed. Try being more persuasive.")

    def get_results(self) -> List[AttackAttempt]:
        return self.results
