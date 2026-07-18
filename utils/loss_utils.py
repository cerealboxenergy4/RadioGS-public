#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from kornia.filters import spatial_gradient
from .image_utils import psnr
import numpy as np
import trimesh
import math


# ---- single-layer ironing (ported from single-layer-surfel Part 1) ----
def scheduled_weight(target_weight, iteration, warmup_iters=0, ramp_iters=0, until_iters=0, decay_iters=0):
    if target_weight <= 0.0:
        return 0.0
    if iteration <= warmup_iters:
        return 0.0
    weight = float(target_weight)
    ramp = 1.0 if ramp_iters <= 0 else min(max((iteration - warmup_iters) / float(ramp_iters), 0.0), 1.0)
    if until_iters is not None and until_iters > 0 and iteration > until_iters:
        if decay_iters <= 0:
            return 0.0
        decay = 1.0 - min(max((iteration - until_iters) / float(decay_iters), 0.0), 1.0)
        weight *= decay
    return weight * ramp


@torch.no_grad()
def gradient_dissent_observation(data_opacity_grad, ironing_opacity_grad, visible,
                                 strength_percentile=0.90, min_pressure_ratio=0.25,
                                 eps=1e-12):
    """Measure per-surfel data-vs-ironing gradient conflict on opacity logits.

    The score is nonzero only when the two objectives request opposite updates and the data
    pressure is not negligible relative to the weighted ironing pressure. Magnitude is normalized
    within the current visible set so the EMA remains comparable across cameras and phases.
    Invisible surfels are returned as zero but must be excluded by the caller's EMA update.
    """
    data = data_opacity_grad.detach().reshape(-1)
    iron = ironing_opacity_grad.detach().reshape(-1)
    visible = visible.detach().reshape(-1).bool()
    if data.shape != iron.shape or data.shape != visible.shape:
        raise ValueError("gradient dissent inputs must have the same per-surfel length")

    score = torch.zeros_like(data)
    if not visible.any():
        return score, {
            "conflict_fraction": score.new_tensor(0.0),
            "strength_scale": score.new_tensor(0.0),
            "score_mean": score.new_tensor(0.0),
        }

    data_abs = data.abs()
    iron_abs = iron.abs()
    percentile = min(max(float(strength_percentile), 0.0), 1.0)
    scale = torch.quantile(data_abs[visible].float(), percentile).to(data.dtype).clamp_min(eps)
    conflict = ((data * iron < 0.0)
                & (data_abs >= float(min_pressure_ratio) * iron_abs)
                & (iron_abs > eps)
                & visible)
    score[conflict] = (data_abs[conflict] / scale).clamp(max=1.0)
    return score, {
        "conflict_fraction": conflict[visible].float().mean(),
        "strength_scale": scale,
        "score_mean": score[visible].mean(),
    }


@torch.no_grad()
def gradient_dissent_gate(dissent_ema, observations, tau=0.25, min_gate=0.10,
                          max_protected_fraction=0.10, min_observations=20):
    """Turn lagged per-surfel dissent into a budgeted multiplier on ironing gradients."""
    ema = dissent_ema.detach().reshape(-1)
    observations = observations.detach().reshape(-1)
    if ema.shape != observations.shape:
        raise ValueError("dissent EMA and observation counts must have the same length")
    gate = torch.ones_like(ema)
    eligible = observations >= int(min_observations)
    n_eligible = int(eligible.sum().item())
    fraction = min(max(float(max_protected_fraction), 0.0), 1.0)
    k = min(n_eligible, int(round(fraction * ema.numel())))
    protected = torch.zeros_like(eligible)
    if k > 0:
        rank = ema.masked_fill(~eligible, -float("inf"))
        protected[torch.topk(rank, k, sorted=False).indices] = True
        floor = min(max(float(min_gate), 0.0), 1.0)
        gate[protected] = floor + (1.0 - floor) * torch.exp(
            -ema[protected] / max(float(tau), 1e-8))
    return gate, protected


@torch.no_grad()
def gradient_dissent_soft_gate(dissent_ema, observations, tau=0.25, min_gate=0.0,
                               min_observations=20, renormalize=False, max_gate=4.0):
    """Continuously price dissent instead of selecting a top-k subset.

    Every sufficiently observed surfel receives ``exp(-EMA / tau)`` consensus
    pressure.  ``renormalize`` restores unit mean pressure (up to
    ``max_gate``), reallocating the reduction from conflicted surfels to
    compliant ones; without it the gate is the intentionally pure relaxation.
    """
    ema = dissent_ema.detach().reshape(-1)
    observations = observations.detach().reshape(-1)
    if ema.shape != observations.shape:
        raise ValueError("dissent EMA and observation counts must have the same length")
    eligible = observations >= int(min_observations)
    floor = min(max(float(min_gate), 0.0), 1.0)
    gate = torch.ones_like(ema)
    gate[eligible] = floor + (1.0 - floor) * torch.exp(
        -ema[eligible] / max(float(tau), 1e-8))
    if renormalize and gate.numel():
        gate = gate / gate.mean().clamp_min(1e-8)
        gate = gate.clamp(max=float(max_gate))
    return gate, eligible & (gate < 1.0)


@torch.no_grad()
def material_gradient_dissent_observation(data_grads, alignment_grads, visible,
                                          learning_rates=None, strength_percentile=0.90,
                                          min_pressure_ratio=0.25, eps=1e-12):
    """Measure data-vs-response-alignment conflict per surfel in Stage 2.

    Unlike :func:`gradient_dissent_observation`, which intentionally observes
    the scalar opacity direction used by Stage-1 N_eff ironing, this function
    combines row gradients for the PBR attributes (base colour, roughness and
    orientation).  Each row is first converted to its optimizer-step scale,
    so the comparison reflects the update that would actually be applied even
    when the attributes use different learning rates.

    ``data_grads`` must come from the reconstruction objective *without* the
    response-alignment term.  ``alignment_grads`` comes from that term alone.
    A surfel dissents only when their aggregate step directions oppose and the
    reconstruction pressure is non-negligible relative to alignment pressure.
    """
    if len(data_grads) != len(alignment_grads):
        raise ValueError("data and alignment gradient lists must have the same length")
    if learning_rates is None:
        learning_rates = [1.0] * len(data_grads)
    if len(learning_rates) != len(data_grads):
        raise ValueError("learning_rates must align with the gradient lists")

    visible = visible.detach().reshape(-1).bool()
    n = visible.numel()
    device = visible.device
    dtype = torch.float32
    data_sq = torch.zeros(n, device=device, dtype=dtype)
    align_sq = torch.zeros_like(data_sq)
    dot = torch.zeros_like(data_sq)

    for data_grad, align_grad, lr in zip(data_grads, alignment_grads, learning_rates):
        if data_grad is None and align_grad is None:
            continue
        template = data_grad if data_grad is not None else align_grad
        if template.shape[0] != n:
            raise ValueError("material dissent gradient lost primitive alignment")
        if data_grad is None:
            data_grad = torch.zeros_like(template)
        if align_grad is None:
            align_grad = torch.zeros_like(template)
        data = data_grad.detach().reshape(n, -1).float() * float(lr)
        align = align_grad.detach().reshape(n, -1).float() * float(lr)
        data_sq.add_(data.square().sum(dim=1))
        align_sq.add_(align.square().sum(dim=1))
        dot.add_((data * align).sum(dim=1))

    score = torch.zeros_like(data_sq)
    if not visible.any():
        return score, {
            "conflict_fraction": score.new_tensor(0.0),
            "strength_scale": score.new_tensor(0.0),
            "score_mean": score.new_tensor(0.0),
        }

    data_norm = data_sq.sqrt()
    align_norm = align_sq.sqrt()
    percentile = min(max(float(strength_percentile), 0.0), 1.0)
    scale = torch.quantile(data_norm[visible], percentile).clamp_min(eps)
    conflict = ((dot < 0.0)
                & (data_norm >= float(min_pressure_ratio) * align_norm)
                & (align_norm > eps)
                & visible)
    score[conflict] = (data_norm[conflict] / scale).clamp(max=1.0)
    return score, {
        "conflict_fraction": conflict[visible].float().mean(),
        "strength_scale": scale,
        "score_mean": score[visible].mean(),
    }


@torch.no_grad()
def single_layer_confidence(gt_image, rend_alpha, rend_dist=None, rend_normal=None,
                            surf_normal=None, mode="hybrid", alpha_threshold=0.5,
                            edge_tau=0.05, dist_scale=1.0, normal_tau=0.15,
                            confidence_floor=0.05, eps=1e-8):
    """Detached confidence map for selective camera-ray ironing.

    Uniform N_eff pressure is least trustworthy at high-frequency appearance edges, silhouettes,
    mixed-depth pixels, and pixels where the accumulated surfel normal disagrees with the
    depth-derived surface normal.  This map protects those locations while retaining pressure on
    smooth, geometrically coherent surface interiors.  The caller uses a confidence-normalized
    weighted mean, so this redistributes (rather than merely weakens) ``lambda_single``.
    """
    mode = str(mode).lower()
    valid_modes = {"none", "edge", "geometry", "hybrid"}
    if mode not in valid_modes:
        raise ValueError("single_confidence_mode must be one of %s, got %r" %
                         (sorted(valid_modes), mode))

    alpha = rend_alpha.detach()
    foreground = alpha > alpha_threshold
    one = torch.ones_like(alpha)
    if mode == "none" or not foreground.any():
        return one, {"mean": one[foreground].mean() if foreground.any() else one.mean()}

    # Soft interior confidence: pixels just over the foreground cutoff are typically silhouettes
    # or partial-coverage pixels, where forcing an exact single layer is ill-conditioned.
    alpha_span = max(1.0 - float(alpha_threshold), eps)
    q_silhouette = ((alpha - float(alpha_threshold)) / alpha_span).clamp(0.0, 1.0)
    components = [q_silhouette]
    stats = {"silhouette": q_silhouette[foreground].mean()}

    if mode in {"edge", "hybrid"}:
        # GT-derived confidence cannot be gamed by the renderer.  Luma Sobel magnitude is in the
        # native [0,1] image range; edge_tau controls the protected-detail bandwidth.
        image = gt_image.detach().clamp(0.0, 1.0)
        luma = (0.299 * image[0:1] + 0.587 * image[1:2] + 0.114 * image[2:3])
        grad = spatial_gradient(luma[None], order=1, normalized=True)[0, 0]
        grad_mag = torch.sqrt(grad.square().sum(dim=0, keepdim=True) + eps)
        q_edge = torch.exp(-grad_mag / max(float(edge_tau), eps))
        components.append(q_edge)
        stats["edge"] = q_edge[foreground].mean()

    if mode in {"geometry", "hybrid"}:
        if rend_dist is None or rend_normal is None or surf_normal is None:
            raise ValueError("geometry/hybrid confidence requires rend_dist, rend_normal, surf_normal")

        # 2DGS distortion directly measures multi-depth mixing.  Normalize by the current view's
        # foreground mean so one scale works across scenes and camera distances.
        distortion = rend_dist.detach().clamp_min(0.0)
        dist_ref = distortion[foreground].mean().clamp_min(eps)
        q_dist = torch.exp(-distortion / (dist_ref * max(float(dist_scale), eps)))

        rn = F.normalize(rend_normal.detach(), dim=0, eps=eps)
        sn = F.normalize(surf_normal.detach(), dim=0, eps=eps)
        normal_disagreement = (1.0 - (rn * sn).sum(dim=0, keepdim=True).clamp(-1.0, 1.0))
        q_normal = torch.exp(-normal_disagreement / max(float(normal_tau), eps))
        components.extend([q_dist, q_normal])
        stats["dist"] = q_dist[foreground].mean()
        stats["normal"] = q_normal[foreground].mean()

    # Geometric mean avoids one noisy cue annihilating the others.  A small floor retains a weak
    # anti-stacking signal even at protected pixels and makes the weighted mean numerically robust.
    confidence = torch.ones_like(alpha)
    for component in components:
        confidence = confidence * component.clamp_min(eps)
    confidence = confidence.pow(1.0 / len(components))
    floor = min(max(float(confidence_floor), 0.0), 1.0)
    confidence = floor + (1.0 - floor) * confidence
    confidence = confidence.detach()
    stats["mean"] = confidence[foreground].mean()
    return confidence, stats


def counterfactual_layer_target(gt_image, full_image, first_hit_image, rend_alpha,
                                first_hit_depth, expected_depth, alpha_threshold=0.5,
                                benefit_tau=0.01, benefit_temperature=0.005,
                                depth_tau=0.02, depth_temperature=0.01,
                                max_layers=2.0, gate_mode="soft", eps=1e-8):
    """Detached adaptive layer target derived from appearance necessity and depth separation.

    A second layer is permitted only where full compositing improves per-pixel RGB L1 over the
    first accepted surfel *and* moves the expected depth away from that first surface.  The depth
    condition prevents texture/material edges at one depth from buying capacity through a stack.
    This is deliberately a local prototype: multi-view persistence and top-2 tail mass are left to
    the next routing stage once the counterfactual signal itself is shown to be useful.
    """
    with torch.no_grad():
        first_error = (first_hit_image.detach() - gt_image.detach()).abs().mean(dim=0, keepdim=True)
        full_error = (full_image.detach() - gt_image.detach()).abs().mean(dim=0, keepdim=True)
        benefit = first_error - full_error
        benefit_gate = torch.sigmoid(
            (benefit - float(benefit_tau)) / max(float(benefit_temperature), eps))

        first_depth = first_hit_depth.detach()
        expected = expected_depth.detach()
        relative_depth_gap = (expected - first_depth).abs() / first_depth.abs().clamp_min(eps)
        depth_gate = torch.sigmoid(
            (relative_depth_gap - float(depth_tau)) / max(float(depth_temperature), eps))

        foreground = rend_alpha.detach() > alpha_threshold
        if gate_mode == "soft":
            gate = benefit_gate * depth_gate
        elif gate_mode == "hard":
            # Routing evidence is detached, so the discontinuous decision cannot be gamed by
            # gradients. This arm directly tests a literal K=1-or-2 capacity exception.
            gate = ((benefit > float(benefit_tau))
                    & (relative_depth_gap > float(depth_tau))).to(benefit_gate.dtype)
        else:
            raise ValueError("gate_mode must be soft or hard, got " + repr(gate_mode))
        gate = gate * foreground.to(benefit_gate.dtype)
        max_layers = min(max(float(max_layers), 1.0), 2.0)
        target = 1.0 + (max_layers - 1.0) * gate
        stats = {
            "target_mean": target[foreground].mean() if foreground.any() else target.new_tensor(1.0),
            "permit_fraction": (gate[foreground] > 0.5).float().mean()
            if foreground.any() else gate.new_tensor(0.0),
            "benefit_mean": benefit[foreground].mean()
            if foreground.any() else benefit.new_tensor(0.0),
            "depth_gap_mean": relative_depth_gap[foreground].mean()
            if foreground.any() else relative_depth_gap.new_tensor(0.0),
        }
    return target.detach(), stats


def single_layer_loss(rend_alpha, rend_alpha_m2, alpha_threshold=0.5, confidence=None,
                      target=None, eps=1e-8):
    """Transmittance-weighted effective layer count N_eff = alpha^2 / sum(w_i^2); penalize (N_eff - 1)
    on foreground pixels.  An optional detached confidence map selectively redistributes the loss
    while preserving its weighted-mean scale. Drives the surfel shell toward one layer per ray."""
    neff = rend_alpha.square() / rend_alpha_m2.clamp_min(eps)
    foreground = rend_alpha > alpha_threshold
    if foreground.any():
        values = neff[foreground]
        if target is None:
            targets = 1.0
        else:
            targets = target.detach().to(device=values.device, dtype=values.dtype)[foreground]
        penalty = (values - targets).clamp_min(0.0)
        if confidence is None:
            loss = penalty.mean()
        else:
            weights = confidence.detach().to(device=penalty.device, dtype=penalty.dtype)[foreground]
            weights = weights.clamp_min(0.0)
            loss = (weights * penalty).sum() / weights.sum().clamp_min(eps)
        return loss, values.detach().mean(), foreground.float().mean()
    zero = rend_alpha.sum() * 0.0
    return zero, zero.detach(), zero.detach()


def keff_loss(trace_alpha, trace_alpha_m2, alpha_threshold=0.5, eps=1e-8):
    """Trace-ray analog of single_layer_loss: per-ray effective layer count along a *traced* ray,
    k_eff = (Sum w_i)^2 / Sum w_i^2 = trace_alpha^2 / trace_alpha_m2; penalize (k_eff - 1) on rays
    that actually hit geometry (trace_alpha > threshold). Drives secondary/GI rays toward a single
    opaque layer per intersection (the part the stage-1 camera-ray N_eff loss can't reach)."""
    keff = trace_alpha.square() / trace_alpha_m2.clamp_min(eps)
    foreground = trace_alpha > alpha_threshold
    if foreground.any():
        values = keff[foreground]
        return (values - 1.0).clamp_min(0.0).mean(), values.detach().mean(), foreground.float().mean()
    zero = trace_alpha.sum() * 0.0
    return zero, zero.detach(), zero.detach()


def alpha_binary_loss(trace_alpha, alpha_threshold=0.5):
    """Opacity binarization on traced secondary rays: penalize a*(1-a) on rays that hit geometry
    (trace_alpha > threshold), pushing each hit ray's composite alpha toward 1. k_eff -> 1 alone is
    not single-layer: under first_hit_only the ray's alpha collapses to its first surfel's alpha, so
    any alpha < 1 leaks environment light through occluded rays and under-scales L_ind. Companion to
    keff_loss on the same ray batch (k_eff -> 1 AND alpha -> 1 makes first-hit tracing near-exact)."""
    foreground = trace_alpha > alpha_threshold
    if foreground.any():
        a = trace_alpha[foreground]
        return (a * (1.0 - a)).mean(), a.detach().mean(), foreground.float().mean()
    zero = trace_alpha.sum() * 0.0
    return zero, zero.detach(), zero.detach()


def response_alignment_loss(viewpoint_camera, render_pkg, opt, iteration):
    """Align PBR attributes only where co-contributing surfels plausibly
    describe the same opaque, locally flat surface.

    The mask excludes silhouettes and true image discontinuities. Stage-2
    geometry uses its standard small ``lr_scale``, allowing normals to adapt
    gradually while material and geometry co-evolve.
    """
    albedo_weight = scheduled_weight(
        opt.lambda_response_align_albedo,
        iteration,
        opt.response_align_warmup_iters,
        opt.response_align_ramp_iters,
        opt.response_align_until_iter,
        opt.response_align_decay_iters,
    )
    roughness_weight = scheduled_weight(
        opt.lambda_response_align_roughness,
        iteration,
        opt.response_align_warmup_iters,
        opt.response_align_ramp_iters,
        opt.response_align_until_iter,
        opt.response_align_decay_iters,
    )
    normal_weight = scheduled_weight(
        opt.lambda_response_align_normal,
        iteration,
        opt.response_align_warmup_iters,
        opt.response_align_ramp_iters,
        opt.response_align_until_iter,
        opt.response_align_decay_iters,
    )

    zero = render_pkg["rend_alpha"].sum() * 0.0
    if albedo_weight <= 0.0 and roughness_weight <= 0.0 and normal_weight <= 0.0:
        return zero, {
            "loss_response_align_albedo": zero.detach(),
            "loss_response_align_roughness": zero.detach(),
            "loss_response_align_normal": zero.detach(),
            "response_align_mask_frac": zero.detach(),
            "response_align_neff": zero.detach(),
        }

    alpha = render_pkg["rend_alpha"]
    neff = render_pkg["rend_neff"]
    mask = (alpha.detach() > opt.response_align_alpha_thresh) & \
           (neff.detach() > opt.response_align_neff_thresh)

    if viewpoint_camera.mask is not None:
        mask = mask & (viewpoint_camera.mask.float().cuda() > 0.5)

    flat_quantile = float(opt.response_align_flat_quantile)
    if 0.0 < flat_quantile < 1.0 and mask.any():
        gt = viewpoint_camera.original_image.cuda()
        gt_grad = spatial_gradient(gt.unsqueeze(0), normalized=False).abs().mean(dim=(1, 2))
        eligible = mask[0]
        threshold = torch.quantile(gt_grad[0][eligible].detach(), flat_quantile)
        mask = mask & (gt_grad <= threshold)

    if not mask.any():
        return zero, {
            "loss_response_align_albedo": zero.detach(),
            "loss_response_align_roughness": zero.detach(),
            "loss_response_align_normal": zero.detach(),
            "response_align_mask_frac": mask.float().mean(),
            "response_align_neff": zero.detach(),
        }

    loss_albedo = render_pkg["response_var_albedo"][mask].mean()
    loss_roughness = render_pkg["response_var_roughness"][mask].mean()
    loss_normal = render_pkg["response_var_normal"][mask].mean()
    loss = (albedo_weight * loss_albedo + roughness_weight * loss_roughness
            + normal_weight * loss_normal)
    stats = {
        "loss_response_align_albedo": loss_albedo.detach(),
        "loss_response_align_roughness": loss_roughness.detach(),
        "loss_response_align_normal": loss_normal.detach(),
        "response_align_mask_frac": mask.float().mean().detach(),
        "response_align_neff": neff[mask].mean().detach(),
        "response_align_weight_albedo": albedo_weight,
        "response_align_weight_roughness": roughness_weight,
        "response_align_weight_normal": normal_weight,
    }
    return loss, stats


def normal_response_alignment_loss(viewpoint_camera, render_pkg, opt, iteration, eps=1e-8):
    """Penalize directional variance among surfel normals on one camera ray.

    ``rend_normal`` is ``sum_i w_i n_i`` and ``rend_alpha`` is ``sum_i w_i``.
    For unit normals, ``1 - ||rend_normal / rend_alpha||^2`` is the weighted
    directional variance. Opaque, multi-contributor, locally flat pixels keep
    the objective away from silhouettes and genuine discontinuities.
    """
    weight = scheduled_weight(
        opt.lambda_response_align_normal,
        iteration,
        opt.response_align_normal_warmup_iters,
        opt.response_align_normal_ramp_iters,
        opt.response_align_normal_until_iter,
        opt.response_align_normal_decay_iters,
    )
    alpha = render_pkg["rend_alpha"]
    zero = alpha.sum() * 0.0
    empty = {
        "loss_response_align_normal": zero.detach(),
        "response_align_normal_mask_frac": zero.detach(),
        "response_align_normal_neff": zero.detach(),
        "response_align_normal_weight": weight,
    }
    if weight <= 0.0:
        return zero, empty

    neff = alpha.square() / render_pkg["rend_alpha_m2"].clamp_min(eps)
    mask = (alpha.detach() > opt.response_align_normal_alpha_thresh) & \
           (neff.detach() > opt.response_align_normal_neff_thresh)
    if viewpoint_camera.mask is not None:
        mask = mask & (viewpoint_camera.mask.float().cuda() > 0.5)

    flat_quantile = float(opt.response_align_normal_flat_quantile)
    if 0.0 < flat_quantile < 1.0 and mask.any():
        gt = viewpoint_camera.original_image.cuda()
        gt_grad = spatial_gradient(gt.unsqueeze(0), normalized=False).abs().mean(dim=(1, 2))
        eligible = mask[0]
        threshold = torch.quantile(gt_grad[0][eligible].detach(), flat_quantile)
        mask = mask & (gt_grad <= threshold)

    if not mask.any():
        empty["response_align_normal_mask_frac"] = mask.float().mean().detach()
        return zero, empty

    mean_normal = render_pkg["rend_normal"] / alpha.clamp_min(eps)
    directional_variance = (1.0 - mean_normal.square().sum(dim=0, keepdim=True)).clamp_min(0.0)
    loss_normal = directional_variance[mask].mean()
    stats = {
        "loss_response_align_normal": loss_normal.detach(),
        "response_align_normal_mask_frac": mask.float().mean().detach(),
        "response_align_normal_neff": neff[mask].mean().detach(),
        "response_align_normal_weight": weight,
    }
    return weight * loss_normal, stats


from utils.graphics_utils import rgb_to_srgb, srgb_to_rgb

def cos_loss(output, gt, thrsh=0, weight=1):
    cos = torch.sum(output * gt * weight, 0)
    return (1 - cos[cos < np.cos(thrsh)]).mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def my_l1_loss(pred, gt):
    return torch.abs((pred - gt))

def my_l2_loss(pred, gt):
    return ((pred - gt) ** 2)

def relMSE(pred, gt, reduction='mean'):
    num = (pred-gt)
    denom = (pred + gt) / 2 + 1e-6
    return (num / denom.detach())**2

def SMAPE(pred, gt):
    denom = (pred + gt + 1e-6)
    return torch.abs(pred - gt) / denom.detach()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()

def tv_loss(depth):
    # return spatial_gradient(data[None], order=2)[0, :, [0, 2]].abs().sum(1).mean()
    h_tv = torch.square(depth[..., 1:, :] - depth[..., :-1, :]).mean()
    w_tv = torch.square(depth[..., :, 1:] - depth[..., :, :-1]).mean()
    return h_tv + w_tv

def calculate_loss(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["rend_alpha"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    visibility_filter = render_pkg["visibility_filter"]
    rend_dist = render_pkg["rend_dist"]
    gt_image = viewpoint_camera.original_image.cuda()

    Ll1 = l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    loss = torch.zeros_like(loss0)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    loss += loss0

    has_sh= (opt.train_sh_vol and iteration <= opt.volume_render_until_iter >= opt.train_sh_vol_from_iter) \
        or (opt.train_sh_surf and iteration > opt.volume_render_until_iter and iteration >= opt.train_sh_surf_from_iter)
    has_sh = has_sh and iteration > opt.init_until_iter

    if has_sh:
        rendered_image_sh = render_pkg["render_sh"]
        loss_sh = (1.0 - opt.lambda_dssim) * l1_loss(rendered_image_sh, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image_sh, gt_image))
        tb_dict["loss_sh"] = loss_sh.item()
        loss += loss_sh

    # Store radiosity tensor for densification gradient computation
    radiosity_tensor = None
    if opt.radiosity and opt.lambda_radiosity >= 0 and has_sh:
        colors_sh = render_pkg["colors_sh"]
        colors_pbr = render_pkg["colors_pbr"]
        loss_radiosity = F.l1_loss(colors_sh, colors_pbr)
        tb_dict["loss_radiosity"] = loss_radiosity.item()
        loss = loss + opt.lambda_radiosity * loss_radiosity
        # Store the tensor for gradient computation in densification
        # Use colors_pbr as it represents the physically-based rendering colors
        radiosity_tensor = torch.abs(colors_pbr - colors_sh).detach()


    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    if opt.lambda_normal_smooth > 0 and iteration > opt.normal_smooth_from_iter and iteration < opt.normal_smooth_until_iter:
        loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)
    
    if viewpoint_camera.mask is not None and opt.lambda_mask_entropy > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        image_mask = viewpoint_camera.mask.float()
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy
    else:
        tb_dict["loss_mask_entropy"] = torch.zeros_like(loss)

    if opt.lambda_light > 0 and 'light' in render_pkg:
        light_direct = render_pkg["light"]
        mean_light = light_direct.mean(-1, keepdim=True).expand_as(light_direct)
        loss_light = F.l1_loss(light_direct, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    if opt.lambda_base_color_smooth > 0:
        rendered_base_color = render_pkg["base_color_linear"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        else:
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color, gt_image)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

    if opt.lambda_roughness_smooth > 0:
        rendered_roughness = render_pkg["roughness_map"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        else:
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness, gt_image)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()

    if opt.lambda_light_smooth > 0:
        env = render_pkg["env_only"]
        loss_light_smooth = tv_loss(env)
        loss = loss + opt.lambda_light_smooth * loss_light_smooth

    try:
        if pc.use_sdf:
            if iteration > 1000:
                ref_dev = pc.get_invs_ref()
                loss_dev = torch.relu(ref_dev - pc.inverse_deviation)
                tb_dict['dev'] = loss_dev
                tb_dict['inv_dev'] = pc.inverse_deviation.mean().item()
                loss = loss + opt.lambda_dev * loss_dev
            if opt.lambda_proj > 0 and iteration > opt.proj_from_iteration:
                points = pc.get_shift_xyz[visibility_filter]
                points = torch.cat([points, torch.ones_like(points[:, -1:])], -1)
                points_view = points @ viewpoint_camera.world_view_transform
                points_proj = points @ viewpoint_camera.full_proj_transform
                points_depth = points_view[:, 2:3]
                uv = points_proj[:, :2] / (points_proj[:, -1:] + 1e-8)
                gaussian_proj_depth = torch.nn.functional.grid_sample(input=rendered_depth.unsqueeze(0),
                                                    grid=uv.view(1, -1, 1, 2),
                                                    mode='bilinear',
                                                    padding_mode='border'  # 'reflection', 'zeros'
                                                    )[0, 0]
                # Detach the projected depth to avoid unstable gradient.
                proj_error = torch.abs(gaussian_proj_depth.detach() - points_depth) 
                loss_proj = (proj_error * (proj_error < opt.proj_thres)).mean()
                tb_dict['proj'] = loss_proj
                loss = loss + opt.lambda_proj * loss_proj
    except:
        pass
    
        
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict, radiosity_tensor

def calculate_loss2(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_normal = render_pkg["rend_normal"]
    gt_image = viewpoint_camera.original_image.cuda()

    rendered_image = render_pkg["render"]
    Ll1 = F.l1_loss(rendered_image, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
    tb_dict["loss_l1"] = Ll1.item()
    loss = Ll1 * opt.lambda_pbr
    
    rendered_image_sh = render_pkg["render_sh"]
    loss_sh = (1.0 - opt.lambda_dssim) * l1_loss(rendered_image_sh, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image_sh, gt_image))
    loss += loss_sh * opt.lambda_nvs

    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        rend_dist = render_pkg["rend_dist"]
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        rendered_depth = render_pkg["surf_depth"]
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)
        
    if viewpoint_camera.mask is not None and opt.lambda_mask_entropy > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        image_mask = viewpoint_camera.mask.float()
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy
    else:
        tb_dict["loss_mask_entropy"] = torch.zeros_like(loss)
    
    if opt.lambda_base_color_smooth > 0:
        rendered_base_color = render_pkg["base_color_linear"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        else:
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color, gt_image)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth
    
    if opt.lambda_metallic_smooth > 0:
        rendered_metallic = render_pkg["metallic"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic * image_mask, gt_image)
        else:
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic, gt_image)
        tb_dict["loss_metallic_smooth"] = loss_metallic_smooth.item()
        loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth
    
    if opt.lambda_roughness_smooth > 0:
        rendered_roughness = render_pkg["roughness"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        else:
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness, gt_image)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth
    
    if opt.lambda_normal_smooth > 0:
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
        else:
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_light > 0:
        light_direct = render_pkg["ray_light_direct"]
        mean_light = light_direct.mean(-1, keepdim=True).expand_as(light_direct)
        loss_light = F.l1_loss(light_direct, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    if opt.lambda_light_smooth > 0:
        env = render_pkg["env_only"]
        loss_light_smooth = tv_loss(env)
        loss = loss + opt.lambda_light_smooth * loss_light_smooth
    
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict

def calculate_loss3(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_normal = render_pkg["rend_normal"]
    gt_image = viewpoint_camera.original_image.cuda()

    rendered_image = render_pkg["render"]
    # Ll1 = (1.0 - opt.lambda_dssim) * l1_loss(rendered_image, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
    Ll1 = l1_loss(rendered_image, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
    tb_dict["loss_l1"] = Ll1.item()
    if iteration > opt.pbr_loss_start:
        loss = Ll1 * opt.lambda_pbr
    else:
        loss = torch.zeros_like(Ll1)
    
    rendered_image_sh = render_pkg["render_sh"]
    loss_sh = (1.0 - opt.lambda_dssim) * l1_loss(rendered_image_sh, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image_sh, gt_image))
    loss += loss_sh * opt.lambda_nvs

    response_loss, response_stats = response_alignment_loss(
        viewpoint_camera, render_pkg, opt, iteration
    )
    loss = loss + response_loss
    tb_dict.update(response_stats)
    # Kept differentiable for Stage-2 material-dissent gradient surgery.  The
    # training loop removes this private entry before TensorBoard logging.
    tb_dict["_response_align_contrib"] = response_loss

    with torch.no_grad():
        rendered_image = render_pkg["render"]
        rendered_image_sh = render_pkg["render_sh"]
        psnr_pbr = psnr(rendered_image, gt_image).mean().item()
        psnr_nvs = psnr(rendered_image_sh, gt_image).mean().item()
        tb_dict["psnr_pbr"] = psnr_pbr
        tb_dict["psnr_nvs"] = psnr_nvs

    if opt.lambda_radiosity >= 0 and iteration > opt.radiosity_loss_start:
        # print('radiosity')
        if opt.rad_loss == 'l1': loss_fn = my_l1_loss
        elif opt.rad_loss == 'l2': loss_fn = my_l2_loss
        elif opt.rad_loss == 'relmse': loss_fn = relMSE
        elif opt.rad_loss == 'smape': loss_fn = SMAPE
        else: raise NotImplementedError("Unknown radiosity loss type", opt.rad_loss)

        if not opt.only_rnd_view:
            pbr_radiosity = render_pkg["pbr_radiosity"]
            nvs_radiosity = render_pkg["nvs_radiosity"]
            loss_radiosity = loss_fn(pbr_radiosity, nvs_radiosity)
        else: loss_radiosity = 0.0

        if 'rand_lhs' in render_pkg.keys() and 'rand_rhs' in render_pkg.keys():
            rand_lhs = render_pkg["rand_lhs"]
            rand_rhs = render_pkg["rand_rhs"]
            rand_loss = loss_fn(rand_lhs, rand_rhs).mean(dim=0)
            if not opt.rad_only_rndview:
                # loss_radiosity += loss_fn(rand_lhs, rand_rhs)
                loss_radiosity += rand_loss
            else:
                # loss_radiosity = loss_fn(rand_lhs, rand_rhs)
                loss_radiosity = rand_loss
            # loss_radiosity = loss_radiosity / 2.0

        if opt.weight_roughness:
            loss_radiosity = loss_radiosity * render_pkg["rad_roughness"]

        loss_radiosity = loss_radiosity.mean()

        tb_dict["loss_radiosity"] = loss_radiosity.item()
        loss = loss + opt.lambda_radiosity * loss_radiosity
        # if opt.rad_only:
        #     tb_dict["loss"] = loss.item()
        #     return loss, tb_dict

        # fhpbr j-consistency: same loss family, applied to the transport-selected first-hit
        # surfels of the subset's secondary rays (see gaussian_renderer/radiogs.py).
        if getattr(opt, 'lambda_fhpbr_j', 0.0) > 0 and 'fhpbr_j_lhs' in render_pkg:
            loss_fhpbr_j = loss_fn(render_pkg['fhpbr_j_lhs'], render_pkg['fhpbr_j_rhs']).mean()
            tb_dict["loss_fhpbr_j"] = loss_fhpbr_j.item()
            loss = loss + opt.lambda_fhpbr_j * loss_fhpbr_j

    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        rend_dist = render_pkg["rend_dist"]
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        rendered_depth = render_pkg["surf_depth"]
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)
        
    if viewpoint_camera.mask is not None and opt.lambda_mask_entropy > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        image_mask = viewpoint_camera.mask.float()
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy
    else:
        tb_dict["loss_mask_entropy"] = torch.zeros_like(loss)
    
    if opt.lambda_base_color_smooth > 0 and iteration > opt.pbr_loss_start:
        rendered_base_color = render_pkg["base_color_linear"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        else:
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color, gt_image)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth
    
    if opt.lambda_metallic_smooth > 0 and iteration > opt.pbr_loss_start:
        rendered_metallic = render_pkg["metallic"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic * image_mask, gt_image)
        else:
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic, gt_image)
        tb_dict["loss_metallic_smooth"] = loss_metallic_smooth.item()
        loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth
    
    if opt.lambda_roughness_smooth > 0 and iteration > opt.pbr_loss_start:
        rendered_roughness = render_pkg["roughness"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        else:
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness, gt_image)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth
    
    if opt.lambda_normal_smooth > 0:
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
        else:
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_light > 0 and iteration > opt.pbr_loss_start:
        light_direct = render_pkg["ray_light_direct"]
        mean_light = light_direct.mean(-1, keepdim=True).expand_as(light_direct)
        loss_light = F.l1_loss(light_direct, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    if opt.lambda_light_smooth > 0 and iteration > opt.pbr_loss_start:
        env = render_pkg["env_only"]
        loss_light_smooth = tv_loss(env)
        loss = loss + opt.lambda_light_smooth * loss_light_smooth
    
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict

def calculate_loss4(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["rend_alpha"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    visibility_filter = render_pkg["visibility_filter"]
    rend_dist = render_pkg["rend_dist"]
    gt_image = viewpoint_camera.original_image.cuda()

    Ll1 = l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    loss = torch.zeros_like(loss0)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    loss += loss0

    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    if opt.lambda_normal_smooth > 0 and iteration > opt.normal_loss_start:
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
        else:
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)
    
    if viewpoint_camera.mask is not None and opt.lambda_mask_entropy > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        image_mask = viewpoint_camera.mask.float()
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy
    else:
        tb_dict["loss_mask_entropy"] = torch.zeros_like(loss)
        
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict


def calculate_radiosity(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    if opt.rad_loss == 'l1': loss_fn = my_l1_loss
    elif opt.rad_loss == 'l2': loss_fn = my_l2_loss
    elif opt.rad_loss == 'relmse': loss_fn = relMSE
    elif opt.rad_loss == 'smape': loss_fn = SMAPE
    else: raise NotImplementedError("Unknown radiosity loss type", opt.rad_loss)

    pbr_radiosity = render_pkg["pbr_radiosity"]
    nvs_radiosity = render_pkg["nvs_radiosity"]
    loss_radiosity = loss_fn(pbr_radiosity, nvs_radiosity)

    if 'rand_lhs' in render_pkg.keys() and 'rand_rhs' in render_pkg.keys():
        rand_lhs = render_pkg["rand_lhs"]
        rand_rhs = render_pkg["rand_rhs"].detach()
        loss_radiosity += loss_fn(rand_lhs, rand_rhs)
        # loss_radiosity = loss_radiosity / 2.0

    if opt.weight_roughness:
        loss_radiosity = loss_radiosity * render_pkg["rad_roughness"]

    loss_radiosity = loss_radiosity.mean()

    tb_dict["loss_radiosity"] = loss_radiosity.item()
    loss = opt.lambda_radiosity * loss_radiosity
    
    tb_dict["loss"] = loss.item()

    render_pbr = render_pkg["render"].detach()
    render_nvs = render_pkg["render_sh"]

    Ll1 = l1_loss(render_nvs, render_pbr)
    ssim_val = ssim(render_nvs, render_pbr)
    loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(render_nvs, render_pbr).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    loss += opt.lambda_nvs * loss0

    return loss, tb_dict
