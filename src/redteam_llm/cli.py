import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from redteam.modules.g2pia_defensive import (
    SAFETY_BANNER,
    evaluate_attack_set,
    load_dataset,
)
from .config import G2PIAConfig, load_config
from .orchestrator import RedTeamOrchestrator
from .types import SafetyLabel

# Ensure local modules (including `redteam.modules`) are importable when running from source.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

console = Console()


@click.group()
def cli():
    """Red Team LLM - Local Safety Evaluation Framework"""
    pass


@cli.command()
@click.option(
    "--config", "-c", default="config.yaml", help="Path to configuration file"
)
@click.option(
    "--output", "-o", default="results.json", help="Path to save results JSON"
)
def run(config, output):
    """Run a red team evaluation session."""
    try:
        cfg = load_config(config)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Config file '{config}' not found.")
        return

    console.print("[bold green]Starting session with:[/bold green]")
    console.print(f"  Target: {cfg.target_model}")
    console.print(f"  Attacker: {cfg.attacker_model}")
    console.print(f"  Judge: {cfg.judge_model}")
    console.print(f"  Strategies: {', '.join(cfg.strategies)}")

    orchestrator = RedTeamOrchestrator(cfg)
    orchestrator.run_session()

    results = orchestrator.get_results()

    # Display Summary
    table = Table(title="Evaluation Summary")
    table.add_column("Strategy", style="cyan")
    table.add_column("Payload", style="magenta")
    table.add_column("Label", style="bold")
    table.add_column("Score", justify="right")

    safe_count = 0
    unsafe_count = 0

    for r in results:
        label_style = (
            "green"
            if r.safety_label == SafetyLabel.SAFE
            else "red"
            if r.safety_label == SafetyLabel.UNSAFE
            else "yellow"
        )
        payload_snippet = (
            (r.input_prompt[:30] + "...")
            if len(r.input_prompt) > 30
            else r.input_prompt
        )
        # Actually display the intended payload if we can infer it, but input_prompt is the full prompt.
        # Let's try to extract the payload or just show the strategy.

        table.add_row(
            r.strategy_name,
            payload_snippet,
            f"[{label_style}]{r.safety_label}[/{label_style}]",
            str(r.safety_score),
        )

        if r.safety_label == SafetyLabel.SAFE:
            safe_count += 1
        elif r.safety_label == SafetyLabel.UNSAFE:
            unsafe_count += 1

    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] {len(results)} | [green]Safe:[/green] {safe_count} | [red]Unsafe:[/red] {unsafe_count}"
    )

    # Save results
    with open(output, "w") as f:
        json.dump([r.dict() for r in results], f, indent=2, default=str)
    console.print(f"Results saved to {output}")


@cli.command("run-g2pia")
@click.option("--dataset", required=True, help="Path to a JSON/JSONL dataset file.")
@click.option(
    "--victim-model",
    required=True,
    help="Victim model name (Ollama or other supported backend).",
)
@click.option(
    "--candidates",
    default=10,
    show_default=True,
    help="Candidates per sample to evaluate.",
)
@click.option(
    "--epsilon", default=None, type=float, help="Semantic distance threshold."
)
@click.option("--gamma", default=None, type=float, help="Target cosine similarity.")
@click.option(
    "--delta", default=None, type=float, help="Cosine tolerance around gamma."
)
@click.option(
    "--max-attempts",
    "max_attempts_opt",
    default=None,
    type=int,
    help="Max generation attempts.",
)
@click.option(
    "--allow-adversarial-sim",
    is_flag=True,
    default=False,
    help="Explicitly enable candidate generation (lab-only).",
)
@click.option(
    "--out",
    default="reports/g2pia_report.json",
    show_default=True,
    help="Path to save JSON report.",
)
@click.option("--html-out", default=None, help="Optional HTML report output path.")
@click.option(
    "--config", "-c", default="config.yaml", help="Framework config for defaults."
)
def run_g2pia(
    dataset,
    victim_model,
    candidates,
    epsilon,
    gamma,
    delta,
    max_attempts_opt,
    allow_adversarial_sim,
    out,
    html_out,
    config,
):
    """Run the defensive G2PIA simulation on a dataset."""
    try:
        cfg = load_config(config)
        g2pia_cfg: G2PIAConfig = cfg.g2pia or G2PIAConfig()
    except FileNotFoundError:
        console.print(
            f"[bold yellow]Config file '{config}' not found; using built-in defaults.[/bold yellow]"
        )
        g2pia_cfg = G2PIAConfig()
    epsilon = epsilon if epsilon is not None else g2pia_cfg.epsilon
    gamma = gamma if gamma is not None else g2pia_cfg.gamma
    delta = delta if delta is not None else g2pia_cfg.delta
    N = g2pia_cfg.N
    max_attempts = (
        max_attempts_opt if max_attempts_opt is not None else g2pia_cfg.max_attempts
    )

    console.print(f"[bold cyan]{SAFETY_BANNER}[/bold cyan]")
    if not allow_adversarial_sim:
        console.print(
            "[bold yellow]--allow-adversarial-sim not set; running clean-only metrics.[/bold yellow]"
        )

    dataset_path = Path(dataset)
    try:
        parse_gsm8k = (
            "gsm8k" in dataset_path.name.lower()
            or "gsm8k" in str(dataset_path.parent.name).lower()
        )
        samples = load_dataset(dataset_path, parse_gsm8k=parse_gsm8k)
    except Exception as exc:
        console.print(f"[bold red]Failed to load dataset:[/bold red] {exc}")
        return

    report = evaluate_attack_set(
        dataset=samples,
        victim_model=victim_model,
        candidates_per_sample=candidates,
        epsilon=epsilon,
        gamma=gamma,
        delta=delta,
        N=N,
        max_attempts=max_attempts,
        allow_adversarial_sim=allow_adversarial_sim,
        generator_model_name=g2pia_cfg.generator_model,
        out_path=Path(out),
        html_out_path=Path(html_out) if html_out else None,
    )

    metrics = report["metrics"]
    console.print(f"[bold green]Run complete[/bold green] -> {out}")
    console.print(f"Clean Accuracy: {metrics['clean_accuracy']:.3f}")
    console.print(f"Attack Accuracy: {metrics['attack_accuracy']:.3f}")
    console.print(f"Attack Success Rate: {metrics['attack_success_rate']:.3f}")
    console.print(f"Confusion buckets: {report['confusion_matrix']}")


if __name__ == "__main__":  # pragma: no cover
    cli()
