"""Bounded reference reconstruction that adjusts only a native timbre token."""
import functools
import math
import time
from types import MethodType

import torch


def install_voice_profile(model, embedding):
    """Attach one execution's learned token through the owning ModelPatcher."""
    clear_voice_profile(model)
    patch_key = "diffusion_model.encoder.timbre_encoder.forward"
    original = model.get_model_object(patch_key)
    previous_patch = model.object_patches.get(patch_key)
    encoder = model.get_model_object("diffusion_model.encoder.timbre_encoder")
    state = {"enabled": False, "embedding": embedding.detach().clone()}

    @functools.wraps(original)
    def profile_forward(self, *args, **kwargs):
        embeddings, mask = original(*args, **kwargs)
        if not state["enabled"]:
            return embeddings, mask
        learned = state["embedding"].to(embeddings)
        return learned.expand_as(embeddings), mask

    profile_forward._acestep_sft_voice_profile = True
    profile_forward._acestep_sft_previous_voice_profile_patch = previous_patch
    model.model_options["_acestep_sft_voice_profile"] = state
    model.add_object_patch(patch_key, MethodType(profile_forward, encoder))


def clear_voice_profile(model):
    state = model.model_options.pop("_acestep_sft_voice_profile", None)
    if state is not None:
        state["enabled"] = False
    patch_key = "diffusion_model.encoder.timbre_encoder.forward"
    current = model.object_patches.get(patch_key)
    if not getattr(current, "_acestep_sft_voice_profile", False):
        return
    previous = current._acestep_sft_previous_voice_profile_patch
    if previous is None:
        model.object_patches.pop(patch_key, None)
    else:
        model.add_object_patch(patch_key, previous)
    encoder = model.get_model_object("diffusion_model.encoder.timbre_encoder")
    if encoder.forward is current:
        model.unpatch_model(unpatch_weights=False)
        model.patch_model(load_weights=False)


def fit_timbre(
    decoder, initial_timbre, examples, *, seconds=60.0, seed=0,
    learning_rate=0.02, regularization=0.05, max_relative_change=0.15,
    progress=None, clock=time.monotonic,
):
    """Fit a small timbre delta, accepting only held-out reconstruction gains.

    Each example contains target [1,T,64], detached encoder lyrics/text, and
    silence-plus-mask context [1,T,128]. The last example is reserved for
    validation when more than one is supplied. Model parameters are never
    optimized. The caller owns loading, frozen weights and the grad context.
    """
    seconds = float(seconds)
    learning_rate = float(learning_rate)
    regularization = float(regularization)
    max_relative_change = float(max_relative_change)
    if not all(math.isfinite(value) for value in (
        seconds, learning_rate, regularization, max_relative_change,
    )):
        raise ValueError("Voice adaptation settings must be finite.")
    if seconds < 0 or learning_rate < 0 or regularization < 0 or max_relative_change < 0:
        raise ValueError("Voice adaptation settings must be nonnegative.")

    report = {
        "accepted": False, "steps": 0, "elapsed_seconds": 0.0,
        "requested_seconds": seconds, "deadline_overrun_seconds": 0.0,
        "initial_validation_loss": None, "final_validation_loss": None,
        "last_training_loss": None, "relative_change": 0.0,
        "validation_held_out": len(examples) > 1,
        "stop_reason": "disabled", "timed_out": False,
    }
    if seconds == 0:
        return initial_timbre, report
    if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
        raise RuntimeError(
            "Voice adaptation requires inference_mode(False) and enable_grad() "
            "around this call; generation may return to inference afterwards."
        )
    if initial_timbre.ndim != 3 or initial_timbre.shape[0] != 1:
        raise ValueError("Voice adaptation expects one [1,tokens,channels] timbre reference.")
    if not examples:
        raise ValueError("Voice adaptation requires at least one reconstruction example.")

    started = clock()
    deadline = started + seconds
    anchor = initial_timbre.detach().float().clone()
    if not bool(torch.isfinite(anchor).all()):
        raise ValueError("Voice adaptation timbre contains NaN or Inf.")
    anchor_rms = anchor.square().mean().sqrt().clamp_min(1e-6)
    prepared = []
    for example in examples:
        current = {name: example[name].detach().clone()
                   for name in ("target", "lyrics", "text", "context")}
        target = current["target"]
        if target.ndim != 3 or target.shape[0] != 1 or target.shape[1] == 0:
            raise ValueError("Voice adaptation target must have shape [1,frames,channels].")
        if current["context"].shape != (*target.shape[:2], 2 * target.shape[-1]):
            raise ValueError("Voice adaptation context must contain silence and a chunk mask.")
        for name, value in current.items():
            if value.device != anchor.device or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Voice adaptation {name} must be finite and on the model device.")
        for name in ("lyrics", "text"):
            if current[name].ndim != 3 or current[name].shape[0] != 1 or current[name].shape[-1] != anchor.shape[-1]:
                raise ValueError(f"Voice adaptation {name} must match the timbre embedding channels.")
        prepared.append(current)

    generator = torch.Generator(device=anchor.device)
    generator.manual_seed(int(seed) % (2 ** 63))
    validation_generator = torch.Generator(device=anchor.device)
    validation_generator.manual_seed((int(seed) + 104729) % (2 ** 63))
    validation = prepared[-1]
    training = prepared[:-1] if len(prepared) > 1 else prepared
    validation_noise = torch.randn(
        validation["target"].shape, device=anchor.device,
        dtype=validation["target"].dtype, generator=validation_generator,
    )
    validation_t = torch.full(
        (1,), 0.5, device=anchor.device, dtype=validation["target"].dtype,
    )

    def reconstruction(embedding, example, noise, timestep):
        target = example["target"]
        t = timestep.reshape(-1, 1, 1)
        states = torch.cat((
            example["lyrics"], embedding.to(example["lyrics"].dtype),
            example["text"],
        ), dim=1)
        predicted = decoder(
            hidden_states=t * noise + (1.0 - t) * target,
            timestep=timestep, timestep_r=timestep,
            attention_mask=None, encoder_hidden_states=states,
            encoder_attention_mask=None, context_latents=example["context"],
        )
        return (predicted.float() - (noise - target).float()).square().mean()

    best = anchor.clone()
    if clock() >= deadline:
        report["stop_reason"] = "budget_before_validation"
    else:
        evaluation_started = clock()
        try:
            initial_loss = float(reconstruction(
                anchor, validation, validation_noise, validation_t,
            ).detach())
        except TimeoutError:
            initial_loss = None
            report["stop_reason"] = "deadline_during_initial_validation"
            report["timed_out"] = True
        validation_cost = max(0.000001, clock() - evaluation_started)
        report["initial_validation_loss"] = initial_loss
        report["final_validation_loss"] = initial_loss
        if initial_loss is None:
            pass
        elif not math.isfinite(initial_loss):
            report["stop_reason"] = "nonfinite_validation"
        else:
            report["stop_reason"] = "budget_exhausted"
            best_loss = initial_loss
            delta = torch.zeros_like(anchor, requires_grad=True)
            first_moment = torch.zeros_like(anchor)
            second_moment = torch.zeros_like(anchor)
            estimated_training_cost = 3.0 * validation_cost
            while True:
                remaining = deadline - clock()
                # Forward/backward kernels cannot be cancelled mid-operation.
                if remaining <= 1.3 * (estimated_training_cost + validation_cost):
                    break
                step_started = clock()
                example = training[report["steps"] % len(training)]
                noise = torch.randn(
                    example["target"].shape, device=anchor.device,
                    dtype=example["target"].dtype, generator=generator,
                )
                timestep = torch.sigmoid(torch.randn(
                    (1,), device=anchor.device,
                    dtype=example["target"].dtype, generator=generator,
                ) - 0.4)
                try:
                    reconstruction_loss = reconstruction(anchor + delta, example, noise, timestep)
                except TimeoutError:
                    report["stop_reason"] = "deadline_during_training"
                    report["timed_out"] = True
                    break
                loss = reconstruction_loss + regularization * (delta / anchor_rms).square().mean()
                if not bool(torch.isfinite(loss)):
                    report["stop_reason"] = "nonfinite_training"
                    break
                if not reconstruction_loss.requires_grad:
                    report["stop_reason"] = "missing_timbre_gradient"
                    break
                try:
                    gradient, = torch.autograd.grad(loss, delta)
                except TimeoutError:
                    report["stop_reason"] = "deadline_during_backward"
                    report["timed_out"] = True
                    break
                except RuntimeError as error:
                    report["stop_reason"] = "backward_failed"
                    report["error"] = str(error)
                    break
                if not bool(torch.isfinite(gradient).all()):
                    report["stop_reason"] = "nonfinite_gradient"
                    break
                if not bool(torch.count_nonzero(gradient)):
                    report["stop_reason"] = "zero_timbre_gradient"
                    break
                count = report["steps"] + 1
                first_moment = 0.9 * first_moment + 0.1 * gradient.detach()
                second_moment = 0.999 * second_moment + 0.001 * gradient.detach().square()
                direction = (first_moment / (1.0 - 0.9 ** count)) / (
                    (second_moment / (1.0 - 0.999 ** count)).sqrt() + 1e-8
                )
                updated = delta.detach() - learning_rate * anchor_rms * direction
                relative_rms = updated.square().mean().sqrt() / anchor_rms
                updated = updated * (max_relative_change / relative_rms.clamp_min(1e-12)).clamp(max=1.0)
                delta = updated.detach().requires_grad_(True)
                report["steps"] = count
                report["last_training_loss"] = float(reconstruction_loss.detach())
                estimated_training_cost = max(0.000001, clock() - step_started)
                del gradient, loss, reconstruction_loss

                if deadline - clock() < 1.3 * validation_cost:
                    report["stop_reason"] = "budget_before_candidate_validation"
                    break
                evaluation_started = clock()
                candidate = (anchor + delta.detach()).detach()
                try:
                    candidate_loss = float(reconstruction(
                        candidate, validation, validation_noise, validation_t,
                    ).detach())
                except TimeoutError:
                    report["stop_reason"] = "deadline_during_candidate_validation"
                    report["timed_out"] = True
                    break
                validation_cost = max(validation_cost, clock() - evaluation_started)
                if math.isfinite(candidate_loss) and candidate_loss < best_loss:
                    best = candidate.clone()
                    best_loss = candidate_loss
                    report["accepted"] = True
                    report["final_validation_loss"] = candidate_loss
                if progress is not None:
                    progress(dict(report))

    report["elapsed_seconds"] = max(0.0, clock() - started)
    report["deadline_overrun_seconds"] = max(0.0, report["elapsed_seconds"] - seconds)
    report["relative_change"] = float((best - anchor).square().mean().sqrt() / anchor_rms)
    return best.to(initial_timbre.dtype).detach(), report
