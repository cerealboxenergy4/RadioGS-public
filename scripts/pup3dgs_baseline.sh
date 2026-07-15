#!/bin/bash
# PUP-3DGS pruning baseline (paper baseline vs single-layer ironing), adapted to 2DGS.
# Mirrors trim2dgs_baseline.sh but replaces trim's densify-on contribution-trimming with
# a faithful PUP prune+recover: continue the stock chkpnt40000 to 47000 in volume mode
# with densification OFF (--densify_until_iter 40000), do ONE Fisher/GN sensitivity prune
# at iter 40100 to the iso-budget target (percent-of-current), then fine-tune to 47000
# -> Stage-2 verbatim from the stock cmd.txt -> eval quartet. Reuses the existing stock
# init checkpoints (no stock re-gen). PUP criterion (arguments/refgs.py + train_init.py):
# per-Gaussian U_g = logdet(sum_pixels g g^T) over spatial params [xyz(3), scaling(2)];
# Hutchinson-estimated; lowest-U_g pruned. See docs/part_2/.
#
# Unlike trim (--densify_until_iter 47000), PUP keeps densify OFF so the count is set
# purely by pruning (trim's densify actually re-inflates the count, e.g. hotdog 49k->88k).
#
# Usage: bash scripts/pup3dgs_baseline.sh [cuda] [phase] [arg3]
# Phases (arg3):
#   tir_hotdog_prune [prune_percent]   |  tir_hotdog_ctrl  (+7k, densify off, NO prune)
#   tir_hotdog_stage2 [tag]            |  tir_hotdog_eval [tag]
#   s4r_<scene>_prune [prune_percent]  |  s4r_<scene>_stage2 [tag]  |  s4r_<scene>_eval [tag]
#   (scenes: air_baloons chair hotdog jugs)

cuda=${1:-4}
phase=${2:?phase required}

# Continuation flags: identical to trim EXCEPT densify OFF (densify_until_iter=40000).
PUP_CONT_FLAGS="--iterations 47000 --volume_render_until_iter 47000 --densify_until_iter 40000 --indirect_from_iter 40001"
# Fisher scoring knobs (see arguments/refgs.py). Single prune round at 40100.
PUP_FISHER="--pup_fisher_draws 4 --pup_fisher_views 30 --pup_fisher_ridge 1e-9"

case $phase in

### ---------------- Phase A: TensoIR hotdog pilot ----------------

tir_hotdog_prune)
# iso-budget default 0.356 (stock ~48.9k -> ironed ~31.5k); override via arg3.
pct=${3:-0.356}
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/TensoIR/hotdog --eval -w \
    -m outputs/part2/radiogs_tir_hotdog_pup3dgs/init \
    --start_checkpoint outputs/part2/radiogs_tir_hotdog_stock/init/chkpnt40000.pth \
    $PUP_CONT_FLAGS \
    --lambda_mask_entropy 0.02 --lambda_dist 1000 --lambda_light 0.01 --lambda_normal_smooth 0.02 \
    --pup_prune --pup_prune_iters 40100 --pup_prune_percent $pct $PUP_FISHER
;;

tir_hotdog_ctrl)
# +7k continuation, densify OFF, NO pruning: isolates the pruning effect from the extra
# 7k recovery iterations (analog of trim's cont7k control).
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/TensoIR/hotdog --eval -w \
    -m outputs/part2/radiogs_tir_hotdog_pup_ctrl/init \
    --start_checkpoint outputs/part2/radiogs_tir_hotdog_stock/init/chkpnt40000.pth \
    $PUP_CONT_FLAGS \
    --lambda_mask_entropy 0.02 --lambda_dist 1000 --lambda_light 0.01 --lambda_normal_smooth 0.02
;;

tir_hotdog_stage2)
# Stage-2 flags verbatim from radiogs_tir_hotdog_stock/radiogs_rr/cmd.txt (same as trim).
tag=${3:-radiogs_tir_hotdog_pup3dgs}
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/TensoIR/hotdog --eval \
    -m outputs/part2/$tag/radiogs --iterations 20000 \
    --start_checkpoint_refgs outputs/part2/$tag/init/chkpnt47000.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --lr_scale 0.01 \
    --lambda_nvs 1.0 --back_culling --lambda_mask_entropy 0.02 --lambda_base_color_smooth 0.2 \
    --lambda_roughness_smooth 0.1 --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 \
    --radiosity_sample_num 64 --use_rad_rndview --detach_rad_global --init_roughness_value 0.6 \
    --lambda_light 0.01 --lambda_light_smooth 0.02 --light_t_min 0.1
;;

tir_hotdog_eval)
# Eval quartet; relighting restricted to bridge to match stock/trim comparators.
tag=${3:-radiogs_tir_hotdog_pup3dgs}
CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/part2/$tag/radiogs --eval --skip_train \
    --diffuse_sample_num 64 --light_t_min 0.1 --back_culling
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_tensoir.py -m outputs/part2/$tag/radiogs --light_t_min 0.1
CUDA_VISIBLE_DEVICES=$cuda python eval_material_tensoir.py -m outputs/part2/$tag/radiogs \
    --albedo_rescale 4 --light_t_min 0.1
CUDA_VISIBLE_DEVICES=$cuda python eval_relighting_tensoir.py -m outputs/part2/$tag/radiogs \
    --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 2 -e light --light_t_min 0.1 \
    --envmaps bridge
;;

### ---------------- Phase A2: TensoIR other scenes (armadillo, ficus, lego) ----------------
# Iso-budget prune percents from stock->ironed count (external/RadioGS ironed chkpnt40000):
#   armadillo 119012->91800 (0.2286), ficus 142587->59625 (0.5818), lego 204213->99055 (0.5149).
# Stage-2 flags verbatim from each scene's radiogs_tir_<scene>_stock/radiogs/cmd.txt; eval across
# ALL 5 relight envmaps (empty --envmaps = all), no --light_t_min (that was hotdog-only).

tir_armadillo_prune|tir_ficus_prune|tir_lego_prune)
scene=${phase#tir_}; scene=${scene%_prune}
case $scene in armadillo) dpct=0.2286 ;; ficus) dpct=0.5818 ;; lego) dpct=0.5149 ;; esac
pct=${3:-$dpct}
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/TensoIR/$scene --eval -w \
    -m outputs/part2/radiogs_tir_${scene}_pup3dgs/init \
    --start_checkpoint outputs/part2/radiogs_tir_${scene}_stock/init/chkpnt40000.pth \
    $PUP_CONT_FLAGS \
    --lambda_mask_entropy 0.02 --lambda_dist 1000 --lambda_light 0.01 --lambda_normal_smooth 0.02 \
    --pup_prune --pup_prune_iters 40100 --pup_prune_percent $pct $PUP_FISHER
;;

tir_armadillo_stage2|tir_ficus_stage2|tir_lego_stage2)
scene=${phase#tir_}; scene=${scene%_stage2}
tag=${3:-radiogs_tir_${scene}_pup3dgs}
case $scene in
  armadillo) extra="--init_roughness_value 0.6 --lambda_light 0.01 --lambda_light_smooth 0.02" ;;
  ficus)     extra="--init_roughness_value 0.6 --lambda_light 0.01 --lambda_light_smooth 0.002" ;;
  lego)      extra="--init_roughness_value 0.8 --lambda_light 0.1 --lambda_light_smooth 0.02" ;;
esac
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/TensoIR/$scene --eval \
    -m outputs/part2/$tag/radiogs --iterations 20000 \
    --start_checkpoint_refgs outputs/part2/$tag/init/chkpnt47000.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --lr_scale 0.01 \
    --lambda_nvs 1.0 --back_culling --lambda_mask_entropy 0.02 --lambda_base_color_smooth 0.2 \
    --lambda_roughness_smooth 0.1 --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 \
    --radiosity_sample_num 64 --use_rad_rndview --detach_rad_global $extra
;;

tir_armadillo_eval|tir_ficus_eval|tir_lego_eval)
scene=${phase#tir_}; scene=${scene%_eval}
tag=${3:-radiogs_tir_${scene}_pup3dgs}
CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/part2/$tag/radiogs --eval --skip_train \
    --diffuse_sample_num 64 --back_culling
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_tensoir.py -m outputs/part2/$tag/radiogs
CUDA_VISIBLE_DEVICES=$cuda python eval_material_tensoir.py -m outputs/part2/$tag/radiogs --albedo_rescale 2
CUDA_VISIBLE_DEVICES=$cuda python eval_relighting_tensoir.py -m outputs/part2/$tag/radiogs \
    --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 2 -e light
;;

### ---------------- Phase B: Synthetic4Relight sweep ----------------
# Reuses the *_s4r_stock_v3 init checkpoints (chkpnt40000) produced by trim2dgs_baseline.sh.

s4r_air_baloons_prune|s4r_hotdog_prune)
scene=${phase#s4r_}; scene=${scene%_prune}; pct=${3:-0.356}
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/Synthetic4Relight/$scene --eval -w \
    -m outputs/part2/radiogs_${scene}_s4r_pup3dgs/init \
    --start_checkpoint outputs/part2/radiogs_${scene}_s4r_stock_v3/init/chkpnt40000.pth \
    $PUP_CONT_FLAGS \
    --lambda_mask_entropy 0.05 --lambda_dist 1000 --lambda_light 0.01 --lambda_normal_smooth 0.02 \
    --pup_prune --pup_prune_iters 40100 --pup_prune_percent $pct $PUP_FISHER
;;

s4r_chair_prune|s4r_jugs_prune)
scene=${phase#s4r_}; scene=${scene%_prune}; pct=${3:-0.356}
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/Synthetic4Relight/$scene --eval -w \
    -m outputs/part2/radiogs_${scene}_s4r_pup3dgs/init \
    --start_checkpoint outputs/part2/radiogs_${scene}_s4r_stock_v3/init/chkpnt40000.pth \
    $PUP_CONT_FLAGS \
    --lambda_mask_entropy 0.05 --lambda_dist 1000 --lambda_light 0.01 --lambda_normal_smooth 0.02 \
    --train_sh_vol --lambda_radiosity 0.1 --radiosity \
    --pup_prune --pup_prune_iters 40100 --pup_prune_percent $pct $PUP_FISHER
;;

s4r_*_stage2)   # flags verbatim from radiogs_<scene>_s4r_stock_v2/radiogs/cmd.txt (same as trim)
scene=${phase#s4r_}; scene=${scene%_stage2}
tag=${3:-radiogs_${scene}_s4r_pup3dgs}
case $scene in
  air_baloons) extra="--lr_scale 0.0 --light_t_min 0.1 --init_roughness_value 0.6" ;;
  chair)       extra="--lr_scale 0.001 --init_roughness_value 0.6" ;;
  hotdog)      extra="--lr_scale 0.001 --light_t_min 0.1 --init_roughness_value 0.6" ;;
  jugs)        extra="--lr_scale 0.001 --light_t_min 0.1 --init_roughness_value 0.8" ;;
esac
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/Synthetic4Relight/$scene --eval \
    -m outputs/part2/$tag/radiogs --iterations 20000 \
    --start_checkpoint_refgs outputs/part2/$tag/init/chkpnt47000.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 \
    --lambda_base_color_smooth 1.0 --lambda_roughness_smooth 0.5 --lambda_light_smooth 0.02 --lambda_light 0.1 \
    --lambda_nvs 1.0 --back_culling \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global $extra
;;

s4r_*_eval)
scene=${phase#s4r_}; scene=${scene%_eval}
tag=${3:-radiogs_${scene}_s4r_pup3dgs}
rescale=2; [ "$scene" = "air_baloons" ] && rescale=1
CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/part2/$tag/radiogs --eval --skip_train \
    --diffuse_sample_num 64 --back_culling
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_syn4.py -m outputs/part2/$tag/radiogs --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_material_syn4.py -m outputs/part2/$tag/radiogs --albedo_rescale $rescale --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_relighting_syn4.py -m outputs/part2/$tag/radiogs \
    --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale $rescale -e light --back_culling
;;

*) echo "unknown phase: $phase"; exit 1 ;;
esac
