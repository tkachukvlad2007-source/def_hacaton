#!/bin/bash
# Фінальний прогноз для задачі 2 у форматі організаторів (inference/*.png поруч із frames/).
#   bash make_inference.sh /шлях/до/папки_з_seqN         (моделі з /data/seg_mc/runs за замовчуванням)
#   MODELS="a.pt b.pt" bash make_inference.sh /шлях      (свої моделі)
# Після цього: cd /шлях && python eval.py
ROOT=${1:?вкажіть папку з seqN/frames}
R=/data/seg_mc/runs
MODELS=${MODELS:-"$R/mc_mbv2_704_pseudo.pt $R/mc_mbv2_512.pt $R/mc_unet_effb4_704.pt $R/mc_segformer_b2_704_bin.pt"}
cd "$(dirname "$0")"
python predict_mc.py "$ROOT" --models $MODELS --smooth 0.5 --obstacle-scale 0.6 --min-obstacle 1000 --keep-tall 0.3 "${@:2}"
