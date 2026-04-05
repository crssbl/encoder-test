from __future__ import annotations

import atexit
import multiprocessing as mp
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor

from data.augment import random_symmetry_transform
from data.generators import sample_batch
from data.polynet_loader import FixedValidationDataset, load_polynet_cvrp_dataset, make_fixed_subset
from eval.metrics import save_learning_curves, save_records_and_artifacts
from eval.profile_memory import get_peak_memory_mb, reset_peak_memory
from models.encoder import EncoderFirstModel
from models.energy import score_candidates
from training.candidate_sampler import build_candidates_from_edge_scores, build_candidates_from_order_logits
from training.consistency import augmentation_consistency_loss
from training.diversity import edge_disagreement
from training.listwise_loss import instance_relative_listwise_loss
from utils.logging import ensure_dir, save_json, save_markdown, setup_logger
from utils.reproducibility import make_torch_generator


class EncoderFirstTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.seed = int(config["seed"])
        self.generator = make_torch_generator(self.seed)

        self.output_dir = Path(config["output_root"]) / config["experiment_name"]
        self.raw_dir = ensure_dir(self.output_dir / "raw")
        self.tables_dir = ensure_dir(self.output_dir / "tables")
        self.figs_dir = ensure_dir(self.output_dir / "figs")
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.logger = setup_logger(self.output_dir)
        self.logger.info("Initializing experiment at %s", self.output_dir)
        self.logger.info("Full config:\n%s", pformat(config, width=120))

        use_cuda = bool(config["device"]["use_cuda"]) and torch.cuda.is_available()
        requested_device_ids = list(config["device"].get("devices", [0]))
        visible_device_count = torch.cuda.device_count() if use_cuda else 0
        available_device_ids = [device_id for device_id in requested_device_ids if device_id < visible_device_count]
        if use_cuda and not available_device_ids:
            available_device_ids = [0]
        self.device_ids = available_device_ids if use_cuda else []
        self.primary_device = torch.device(f"cuda:{self.device_ids[0]}" if use_cuda else "cpu")
        self.use_amp = bool(config["device"]["amp"]) and use_cuda
        if use_cuda:
            torch.cuda.set_device(self.primary_device)
        self.logger.info(
            "CUDA requested=%s visible_count=%d using_device_ids=%s primary=%s",
            requested_device_ids,
            visible_device_count,
            self.device_ids,
            self.primary_device,
        )
        self.projector_executors: dict[str, ProcessPoolExecutor] = {}
        self._setup_projector_executors()

        base_model = EncoderFirstModel(config["model"], density_k=config["data"]["density_k"])
        base_model.to(self.primary_device)
        if use_cuda and len(self.device_ids) > 1:
            self.model: nn.Module = nn.DataParallel(base_model, device_ids=self.device_ids)
        else:
            self.model = base_model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["training"]["lr"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history: List[Dict[str, float]] = []
        self.validation_history: List[Dict[str, object]] = []
        self.eval_records: List[Dict[str, object]] = []
        self.examples: List[Dict[str, object]] = []
        self.fixed_validation_full: FixedValidationDataset | None = None
        self.fixed_validation_small: FixedValidationDataset | None = None
        self._setup_fixed_validation()

    def _autocast_context(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _move_batch(self, batch: Dict[str, object]) -> Dict[str, object]:
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.primary_device)
            else:
                moved[key] = value
        return moved

    def _reset_memory(self) -> None:
        reset_peak_memory(range(torch.cuda.device_count())) if torch.cuda.is_available() else None

    def _setup_fixed_validation(self) -> None:
        validation_cfg = self.config.get("validation", {})
        if not validation_cfg.get("enabled", False):
            self.logger.info("Fixed validation disabled in config.")
            return

        dataset_cfg = validation_cfg.get("dataset", {})
        dataset_path = dataset_cfg.get("path")
        if not dataset_path:
            raise ValueError("validation.enabled=True but validation.dataset.path is missing")

        self.fixed_validation_full = load_polynet_cvrp_dataset(dataset_path)
        subset_size = int(validation_cfg.get("subset_size", len(self.fixed_validation_full)))
        subset_seed = int(validation_cfg.get("subset_seed", self.seed + 101))
        self.fixed_validation_small = make_fixed_subset(self.fixed_validation_full, subset_size=subset_size, seed=subset_seed)

        self.logger.info(
            "Loaded fixed validation dataset from %s with %d instances; small subset has %d instances.",
            dataset_path,
            len(self.fixed_validation_full),
            len(self.fixed_validation_small),
        )

    def _setup_projector_executors(self) -> None:
        projector_cfg = self.config.get("projector", {})
        backend = projector_cfg.get("backend", "process")
        if backend != "process":
            self.logger.info("Projector executor backend=%s; using inline execution.", backend)
            return

        requested_scopes = {
            "train": int(self.config["training"].get("projector_workers", 1)),
            "validation_small": int(self.config.get("validation", {}).get("small_projector_workers", 1)),
            "validation_full": int(self.config.get("validation", {}).get("full_projector_workers", 1)),
            "evaluation": int(self.config.get("evaluation", {}).get("projector_workers", 1)),
        }
        ctx = mp.get_context(projector_cfg.get("start_method", "fork"))
        for scope, workers in requested_scopes.items():
            if workers > 1:
                self.projector_executors[scope] = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
                self.logger.info("Created projector executor scope=%s workers=%d backend=process", scope, workers)

        if self.projector_executors:
            atexit.register(self._shutdown_projector_executors)

    def _shutdown_projector_executors(self) -> None:
        for scope, executor in self.projector_executors.items():
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        self.projector_executors.clear()

    def _projector_worker_count(self, scope: str) -> int:
        if scope == "train":
            return int(self.config["training"].get("projector_workers", 1))
        if scope == "validation_small":
            return int(self.config.get("validation", {}).get("small_projector_workers", 1))
        if scope == "validation_full":
            return int(self.config.get("validation", {}).get("full_projector_workers", 1))
        if scope == "evaluation":
            return int(self.config.get("evaluation", {}).get("projector_workers", 1))
        return 1

    def _projector_executor(self, scope: str) -> ProcessPoolExecutor | None:
        return self.projector_executors.get(scope)

    def _train_step(self, step_idx: int) -> Dict[str, float]:
        self.model.train()
        batch = sample_batch(
            distribution=self.config["data"]["train_distribution"],
            batch_size=self.config["data"]["train_batch_size"],
            problem_size=self.config["data"]["train_problem_size"],
            generator=self.generator,
        )
        batch_gpu = self._move_batch(batch)
        self._reset_memory()
        start_time = time.perf_counter()

        with self._autocast_context():
            edge_scores, _, _, _ = self.model(
                batch_gpu["depot"],
                batch_gpu["coords"],
                batch_gpu["demands"],
                compute_order_logits=False,
            )
            consistency_loss = torch.zeros((), device=self.primary_device)
            if float(self.config["training"]["lambda_cons"]) > 0.0:
                aug_depot, aug_coords = random_symmetry_transform(batch["depot"], batch["coords"], self.generator)
                aug_depot = aug_depot.to(self.primary_device)
                aug_coords = aug_coords.to(self.primary_device)
                aug_scores, _, _, _ = self.model(
                    aug_depot,
                    aug_coords,
                    batch_gpu["demands"],
                    compute_order_logits=False,
                )
                consistency_loss = augmentation_consistency_loss(edge_scores, aug_scores)

        candidate_bundle = build_candidates_from_edge_scores(
            edge_scores=edge_scores,
            coords=batch_gpu["coords"],
            depot_xy=batch_gpu["depot"],
            demands=batch_gpu["demands"],
            num_candidates=self.config["training"]["num_candidates"],
            noise_scale=float(self.config["training"]["noise_scale"]),
            capacity=float(self.config["data"]["capacity"]),
            anchor_strategy=self.config["training"]["anchor_strategy"],
            generator=self.generator,
            include_greedy_candidate=bool(self.config["training"]["include_greedy_candidate"]),
            num_workers=self._projector_worker_count("train"),
            executor=self._projector_executor("train"),
        )
        costs = candidate_bundle["costs"].to(self.primary_device)
        successors = candidate_bundle["successors"].to(self.primary_device)
        energies = score_candidates(edge_scores, successors)
        loss_bundle = instance_relative_listwise_loss(
            costs=costs,
            energies=energies,
            tau_cost=float(self.config["training"]["tau_cost"]),
            tau_energy=float(self.config["training"]["tau_energy"]),
        )
        diversity = edge_disagreement(successors)
        total_loss = loss_bundle["loss"]
        total_loss = total_loss + float(self.config["training"]["lambda_div"]) * (-diversity)
        total_loss = total_loss + float(self.config["training"]["lambda_cons"]) * consistency_loss

        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["training"]["grad_clip_norm"]))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["training"]["grad_clip_norm"]))
            self.optimizer.step()

        elapsed = time.perf_counter() - start_time
        peak_memory = get_peak_memory_mb(range(torch.cuda.device_count()))
        greedy_cost = float(costs[:, 0].mean().item())
        best_cost = float(costs.min(dim=-1).values.mean().item())
        metrics = {
            "step": float(step_idx),
            "loss": float(total_loss.item()),
            "listwise_loss": float(loss_bundle["loss"].item()),
            "consistency_loss": float(consistency_loss.item()),
            "edge_disagreement": float(diversity.item()),
            "greedy_cost": greedy_cost,
            "best_cost": best_cost,
            "elapsed_sec": elapsed,
            "peak_memory_mb": peak_memory,
        }
        return metrics

    def train(self) -> None:
        train_steps = int(self.config["data"]["train_steps"])
        if train_steps <= 0:
            self.logger.info("Skipping training because train_steps=0.")
            return

        self.logger.info("Starting training for %d steps", train_steps)
        for step_idx in range(1, train_steps + 1):
            metrics = self._train_step(step_idx)
            self.history.append(metrics)
            should_save_state = False
            if step_idx % int(self.config["training"]["log_every"]) == 0:
                self.logger.info(
                    "step=%d loss=%.4f greedy=%.4f best=%.4f div=%.4f mem=%.1fMB time=%.2fs",
                    step_idx,
                    metrics["loss"],
                    metrics["greedy_cost"],
                    metrics["best_cost"],
                    metrics["edge_disagreement"],
                    metrics["peak_memory_mb"],
                    metrics["elapsed_sec"],
                )
                should_save_state = True
            if step_idx % int(self.config["training"]["checkpoint_every"]) == 0 or step_idx == train_steps:
                self.save_checkpoint(step_idx)
                should_save_state = True
            ran_validation = self._maybe_run_fixed_validation(step_idx)
            should_save_state = should_save_state or ran_validation
            if should_save_state:
                self._save_learning_state()

        self._save_learning_state()

    def save_checkpoint(self, step_idx: int) -> Path:
        payload = {
            "config": self.config,
            "model_state_dict": self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step_idx,
            "history": self.history,
            "validation_history": self.validation_history,
        }
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step_idx}.pt"
        torch.save(payload, checkpoint_path)
        torch.save(payload, self.checkpoint_dir / "latest.pt")
        return checkpoint_path

    def load_latest_checkpoint(self) -> None:
        checkpoint_path = self.checkpoint_dir / "latest.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.primary_device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", [])
        self.validation_history = checkpoint.get("validation_history", [])

    def _save_learning_state(self) -> None:
        if self.history:
            pd.DataFrame(self.history).to_csv(self.raw_dir / "train_history.csv", index=False)
        if self.validation_history:
            pd.DataFrame(self.validation_history).to_csv(self.raw_dir / "validation_history.csv", index=False)
        save_learning_curves(self.history, self.validation_history, self.output_dir)

    def _maybe_run_fixed_validation(self, step_idx: int) -> bool:
        validation_cfg = self.config.get("validation", {})
        if not validation_cfg.get("enabled", False) or self.fixed_validation_full is None or self.fixed_validation_small is None:
            return False

        small_every = int(validation_cfg.get("small_every", 0))
        full_every = int(validation_cfg.get("full_every", 0))
        ran_validation = False

        if small_every > 0 and step_idx % small_every == 0:
            self._run_fixed_validation(step_idx=step_idx, scope="small")
            ran_validation = True
        if full_every > 0 and step_idx % full_every == 0:
            self._run_fixed_validation(step_idx=step_idx, scope="full")
            ran_validation = True
        return ran_validation

    def _run_fixed_validation(self, step_idx: int, scope: str) -> None:
        validation_cfg = self.config["validation"]
        if scope == "small":
            dataset = self.fixed_validation_small
            num_candidates = int(validation_cfg.get("small_num_candidates", self.config["training"]["eval_num_candidates"]))
            methods = validation_cfg.get("small_methods", ["edge_field_greedy", "edge_field_best_of_m"])
            worker_scope = "validation_small"
        elif scope == "full":
            dataset = self.fixed_validation_full
            num_candidates = int(validation_cfg.get("full_num_candidates", self.config["training"]["eval_num_candidates"]))
            methods = validation_cfg.get("full_methods", ["edge_field_greedy", "edge_field_best_of_m"])
            worker_scope = "validation_full"
        else:
            raise ValueError(f"Unsupported validation scope: {scope}")

        assert dataset is not None
        batch_size = int(validation_cfg.get("batch_size", self.config["data"]["eval_batch_size"]))
        projector_workers = self._projector_worker_count(worker_scope)
        self.logger.info(
            "Running %s fixed validation at step=%d on %d instances with methods=%s, candidates=%d, workers=%d",
            scope,
            step_idx,
            len(dataset),
            methods,
            num_candidates,
            projector_workers,
        )

        was_training = self.model.training
        self.model.eval()
        method_records: dict[str, list[dict[str, object]]] = {method: [] for method in methods}

        with torch.inference_mode():
            for batch in dataset.iter_batches(batch_size):
                batch_gpu = self._move_batch(batch)
                needs_order_logits = any(method.startswith("order_sort") for method in methods)
                with self._autocast_context():
                    edge_scores, _, _, order_logits = self.model(
                        batch_gpu["depot"],
                        batch_gpu["coords"],
                        batch_gpu["demands"],
                        compute_order_logits=needs_order_logits,
                    )

                for method_name in methods:
                    self._reset_memory()
                    start = time.perf_counter()
                    if method_name == "edge_field_greedy":
                        bundle = build_candidates_from_edge_scores(
                            edge_scores=edge_scores,
                            coords=batch_gpu["coords"],
                            depot_xy=batch_gpu["depot"],
                            demands=batch_gpu["demands"],
                            num_candidates=1,
                            noise_scale=0.0,
                            capacity=float(self.config["data"]["capacity"]),
                            anchor_strategy=self.config["training"]["anchor_strategy"],
                            generator=self.generator,
                            include_greedy_candidate=True,
                            num_workers=projector_workers,
                            executor=self._projector_executor(worker_scope),
                        )
                        candidate_count = 1
                    elif method_name == "edge_field_best_of_m":
                        bundle = build_candidates_from_edge_scores(
                            edge_scores=edge_scores,
                            coords=batch_gpu["coords"],
                            depot_xy=batch_gpu["depot"],
                            demands=batch_gpu["demands"],
                            num_candidates=num_candidates,
                            noise_scale=float(self.config["training"]["noise_scale"]),
                            capacity=float(self.config["data"]["capacity"]),
                            anchor_strategy=self.config["training"]["anchor_strategy"],
                            generator=self.generator,
                            include_greedy_candidate=True,
                            num_workers=projector_workers,
                            executor=self._projector_executor(worker_scope),
                        )
                        candidate_count = num_candidates
                    elif method_name == "order_sort_greedy":
                        bundle = build_candidates_from_order_logits(
                            order_logits=order_logits,
                            coords=batch_gpu["coords"],
                            depot_xy=batch_gpu["depot"],
                            demands=batch_gpu["demands"],
                            num_candidates=1,
                            noise_scale=0.0,
                            capacity=float(self.config["data"]["capacity"]),
                            generator=self.generator,
                            include_greedy_candidate=True,
                            num_workers=projector_workers,
                            executor=self._projector_executor(worker_scope),
                        )
                        candidate_count = 1
                    elif method_name == "order_sort_best_of_m":
                        bundle = build_candidates_from_order_logits(
                            order_logits=order_logits,
                            coords=batch_gpu["coords"],
                            depot_xy=batch_gpu["depot"],
                            demands=batch_gpu["demands"],
                            num_candidates=num_candidates,
                            noise_scale=float(self.config["training"]["noise_scale"]),
                            capacity=float(self.config["data"]["capacity"]),
                            generator=self.generator,
                            include_greedy_candidate=True,
                            num_workers=projector_workers,
                            executor=self._projector_executor(worker_scope),
                        )
                        candidate_count = num_candidates
                    else:
                        raise ValueError(f"Unsupported validation method: {method_name}")

                    wall_time = time.perf_counter() - start
                    peak_memory = get_peak_memory_mb(range(torch.cuda.device_count()))
                    record = self._evaluate_method(
                        method_name=method_name,
                        bundle=bundle,
                        candidate_count=candidate_count,
                        wall_time_sec=wall_time,
                        peak_memory_mb=peak_memory,
                        distribution="fixed_validation",
                        problem_size=int(batch["problem_size"]),
                    )
                    method_records[method_name].append(record)

        if was_training:
            self.model.train()

        for method_name, records in method_records.items():
            if not records:
                continue
            record_df = pd.DataFrame(records)
            summary_record = record_df.mean(numeric_only=True).to_dict() | {
                "distribution": "fixed_validation",
                "problem_size": int(dataset.coords.size(1)),
                "method": method_name,
                "scope": scope,
                "step": step_idx,
                "dataset_size": len(dataset),
            }
            self.validation_history.append(summary_record)
            self.logger.info(
                "validation scope=%s step=%d method=%s avg_cost=%.4f feas=%.4f avg_routes=%.2f wall=%.2fs",
                scope,
                step_idx,
                method_name,
                summary_record["avg_cost"],
                summary_record["feasibility_rate"],
                summary_record["avg_route_count"],
                summary_record["wall_time_sec"],
            )

    def _evaluate_method(
        self,
        method_name: str,
        bundle: Dict[str, object],
        candidate_count: int,
        wall_time_sec: float,
        peak_memory_mb: float,
        distribution: str,
        problem_size: int,
    ) -> Dict[str, object]:
        costs = bundle["costs"]
        route_counts = bundle["route_counts"]
        split_feasible = bundle["split_feasible"].float()
        projector_valid = bundle["projector_valid"].float()
        successors = bundle["successors"]

        greedy_cost = costs[:, 0]
        best_cost = costs.min(dim=-1).values
        improvement = (greedy_cost - best_cost).mean().item()
        disagreement = edge_disagreement(successors).item()

        record = {
            "distribution": distribution,
            "problem_size": problem_size,
            "method": method_name,
            "candidate_count": candidate_count,
            "avg_cost": float(best_cost.mean().item()) if "best_of_m" in method_name else float(greedy_cost.mean().item()),
            "avg_route_count": float(route_counts.float().mean().item()),
            "feasibility_rate": float(split_feasible.mean().item()),
            "projector_validity_rate": float(projector_valid.mean().item()),
            "avg_edge_disagreement": float(disagreement),
            "avg_best_minus_greedy": float(improvement),
            "wall_time_sec": float(wall_time_sec),
            "instances_per_sec": float(costs.size(0) / max(wall_time_sec, 1e-6)),
            "candidates_per_sec": float(costs.numel() / max(wall_time_sec, 1e-6)),
            "peak_memory_mb": float(peak_memory_mb),
        }
        return record

    def _capture_example(
        self,
        payloads: List[List[Dict[str, object]]],
        costs: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        title: str,
    ) -> None:
        if len(self.examples) >= int(self.config["evaluation"]["examples_to_plot"]):
            return
        best_index = int(costs[0].argmin().item())
        payload = payloads[0][best_index]
        self.examples.append(
            {
                "coords": batch["coords"][0].cpu().numpy(),
                "depot": batch["depot"][0, 0].cpu().numpy(),
                "order": payload["order"],
                "routes": payload["routes"],
                "title": title,
            }
        )

    def evaluate(self) -> None:
        self.model.eval()
        eval_batch_size = int(self.config["data"]["eval_batch_size"])
        eval_instances = int(self.config["data"]["eval_instances"])
        eval_num_candidates = int(self.config["training"]["eval_num_candidates"])
        distributions = list(dict.fromkeys(self.config["data"]["eval_distributions"]))
        size_settings = sorted(
            set(self.config["data"]["size_generalization_sizes"] + self.config["data"]["distribution_shift_sizes"])
        )

        with torch.inference_mode():
            for distribution in distributions:
                for problem_size in size_settings:
                    self.logger.info("Evaluating distribution=%s problem_size=%d", distribution, problem_size)
                    aggregate_batches = max(1, eval_instances // eval_batch_size)
                    edge_greedy_records = []
                    edge_best_records = []
                    order_greedy_records = []
                    order_best_records = []
                    no_split_records = []

                    for _ in range(aggregate_batches):
                        batch = sample_batch(distribution, eval_batch_size, problem_size, self.generator)
                        batch_gpu = self._move_batch(batch)
                        self._reset_memory()
                        needs_order_logits = bool(self.config["evaluation"]["run_order_baseline"])

                        with self._autocast_context():
                            edge_scores, _, _, order_logits = self.model(
                                batch_gpu["depot"],
                                batch_gpu["coords"],
                                batch_gpu["demands"],
                                compute_order_logits=needs_order_logits,
                            )

                        start = time.perf_counter()
                        edge_greedy_bundle = build_candidates_from_edge_scores(
                            edge_scores=edge_scores,
                            coords=batch_gpu["coords"],
                            depot_xy=batch_gpu["depot"],
                            demands=batch_gpu["demands"],
                            num_candidates=1,
                            noise_scale=0.0,
                            capacity=float(self.config["data"]["capacity"]),
                            anchor_strategy=self.config["training"]["anchor_strategy"],
                            generator=self.generator,
                            include_greedy_candidate=True,
                            num_workers=self._projector_worker_count("evaluation"),
                            executor=self._projector_executor("evaluation"),
                        )
                        edge_greedy_time = time.perf_counter() - start
                        peak_memory = get_peak_memory_mb(range(torch.cuda.device_count()))
                        edge_greedy_records.append(
                            self._evaluate_method(
                                "edge_field_greedy",
                                edge_greedy_bundle,
                                1,
                                edge_greedy_time,
                                peak_memory,
                                distribution,
                                problem_size,
                            )
                        )

                        start = time.perf_counter()
                        edge_best_bundle = build_candidates_from_edge_scores(
                            edge_scores=edge_scores,
                            coords=batch_gpu["coords"],
                            depot_xy=batch_gpu["depot"],
                            demands=batch_gpu["demands"],
                            num_candidates=eval_num_candidates,
                            noise_scale=float(self.config["training"]["noise_scale"]),
                            capacity=float(self.config["data"]["capacity"]),
                            anchor_strategy=self.config["training"]["anchor_strategy"],
                            generator=self.generator,
                            include_greedy_candidate=True,
                            num_workers=self._projector_worker_count("evaluation"),
                            executor=self._projector_executor("evaluation"),
                        )
                        edge_best_time = time.perf_counter() - start
                        edge_best_records.append(
                            self._evaluate_method(
                                "edge_field_best_of_m",
                                edge_best_bundle,
                                eval_num_candidates,
                                edge_best_time,
                                peak_memory,
                                distribution,
                                problem_size,
                            )
                        )
                        self._capture_example(
                            edge_best_bundle["payloads"],
                            edge_best_bundle["costs"],
                            batch_gpu,
                            f"{distribution}-n{problem_size}-edge",
                        )

                        if bool(self.config["evaluation"]["run_order_baseline"]):
                            start = time.perf_counter()
                            order_greedy_bundle = build_candidates_from_order_logits(
                                order_logits=order_logits,
                                coords=batch_gpu["coords"],
                                depot_xy=batch_gpu["depot"],
                                demands=batch_gpu["demands"],
                                num_candidates=1,
                                noise_scale=0.0,
                                capacity=float(self.config["data"]["capacity"]),
                                generator=self.generator,
                                include_greedy_candidate=True,
                                num_workers=self._projector_worker_count("evaluation"),
                                executor=self._projector_executor("evaluation"),
                            )
                            order_greedy_time = time.perf_counter() - start
                            order_greedy_records.append(
                                self._evaluate_method(
                                    "order_sort_greedy",
                                    order_greedy_bundle,
                                    1,
                                    order_greedy_time,
                                    peak_memory,
                                    distribution,
                                    problem_size,
                                )
                            )

                            start = time.perf_counter()
                            order_best_bundle = build_candidates_from_order_logits(
                                order_logits=order_logits,
                                coords=batch_gpu["coords"],
                                depot_xy=batch_gpu["depot"],
                                demands=batch_gpu["demands"],
                                num_candidates=eval_num_candidates,
                                noise_scale=float(self.config["training"]["noise_scale"]),
                                capacity=float(self.config["data"]["capacity"]),
                                generator=self.generator,
                                include_greedy_candidate=True,
                                num_workers=self._projector_worker_count("evaluation"),
                                executor=self._projector_executor("evaluation"),
                            )
                            order_best_time = time.perf_counter() - start
                            order_best_records.append(
                                self._evaluate_method(
                                    "order_sort_best_of_m",
                                    order_best_bundle,
                                    eval_num_candidates,
                                    order_best_time,
                                    peak_memory,
                                    distribution,
                                    problem_size,
                                )
                            )

                        if bool(self.config["evaluation"]["run_no_split_ablation"]):
                            start = time.perf_counter()
                            no_split_bundle = build_candidates_from_edge_scores(
                                edge_scores=edge_scores,
                                coords=batch_gpu["coords"],
                                depot_xy=batch_gpu["depot"],
                                demands=batch_gpu["demands"],
                                num_candidates=eval_num_candidates,
                                noise_scale=float(self.config["training"]["noise_scale"]),
                                capacity=float(self.config["data"]["capacity"]),
                                anchor_strategy=self.config["training"]["anchor_strategy"],
                                generator=self.generator,
                                include_greedy_candidate=True,
                                use_no_split=True,
                                num_workers=self._projector_worker_count("evaluation"),
                                executor=self._projector_executor("evaluation"),
                            )
                            no_split_time = time.perf_counter() - start
                            no_split_records.append(
                                self._evaluate_method(
                                    "edge_field_no_split_best_of_m",
                                    no_split_bundle,
                                    eval_num_candidates,
                                    no_split_time,
                                    peak_memory,
                                    distribution,
                                    problem_size,
                                )
                            )

                    for record_group in (
                        edge_greedy_records,
                        edge_best_records,
                        order_greedy_records,
                        order_best_records,
                        no_split_records,
                    ):
                        if not record_group:
                            continue
                        record_df = pd.DataFrame(record_group)
                        self.eval_records.append(record_df.mean(numeric_only=True).to_dict() | {
                            "distribution": distribution,
                            "problem_size": problem_size,
                            "method": record_group[0]["method"],
                        })
                    self.logger.info("Finished distribution=%s problem_size=%d", distribution, problem_size)

        pd.DataFrame(self.eval_records).to_csv(self.raw_dir / "eval_records.csv", index=False)
        save_records_and_artifacts(
            records=self.eval_records,
            train_distribution=self.config["data"]["train_distribution"],
            train_problem_size=int(self.config["data"]["train_problem_size"]),
            output_dir=self.output_dir,
            examples=self.examples,
        )

    def write_summary(self) -> None:
        summary = {
            "experiment_name": self.config["experiment_name"],
            "train_steps": int(self.config["data"]["train_steps"]),
            "history_last": self.history[-1] if self.history else {},
            "validation_last": self.validation_history[-1] if self.validation_history else {},
            "num_validation_records": len(self.validation_history),
            "num_eval_records": len(self.eval_records),
        }
        if self.eval_records:
            best_record = min(self.eval_records, key=lambda item: item["avg_cost"])
            summary["best_eval_record"] = best_record
        if self.validation_history:
            best_validation = min(self.validation_history, key=lambda item: item["avg_cost"])
            summary["best_validation_record"] = best_validation
        save_json(summary, self.output_dir / "summary.json")

        markdown_lines = [
            f"# {self.config['experiment_name']}",
            "",
            f"- train_steps: {self.config['data']['train_steps']}",
            f"- validation_records: {len(self.validation_history)}",
            f"- eval_records: {len(self.eval_records)}",
        ]
        if self.history:
            last = self.history[-1]
            markdown_lines.append(f"- last_train_loss: {last['loss']:.4f}")
            markdown_lines.append(f"- last_train_best_cost: {last['best_cost']:.4f}")
        if self.validation_history:
            best_validation = min(self.validation_history, key=lambda item: item["avg_cost"])
            markdown_lines.append(
                f"- best_validation: {best_validation['scope']} {best_validation['method']} step={int(best_validation['step'])} cost={best_validation['avg_cost']:.4f}"
            )
        if self.eval_records:
            best = min(self.eval_records, key=lambda item: item["avg_cost"])
            markdown_lines.append(
                f"- best_eval: {best['method']} on {best['distribution']} n={int(best['problem_size'])} cost={best['avg_cost']:.4f}"
            )
        save_markdown("\n".join(markdown_lines) + "\n", self.output_dir / "summary.md")

    def run(self, eval_only: bool = False) -> None:
        if eval_only:
            self.load_latest_checkpoint()
        else:
            self.train()
        if bool(self.config.get("evaluation", {}).get("enabled", True)):
            self.evaluate()
        else:
            self.logger.info("Skipping synthetic post-training evaluation because evaluation.enabled is false.")
        self._save_learning_state()
        self.write_summary()
