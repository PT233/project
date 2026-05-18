#!/bin/bash
# Wait for DLBCL training to finish, then run ablation and DLBCL evaluation

PROJECT=/mnt/h/1.MasterDegree/17.RD/project
LOG=$PROJECT/pipeline.log
cd $PROJECT

echo "=== Pipeline started: $(date) ===" >> $LOG

# Wait for DLBCL training process to finish
echo "Waiting for DLBCL training (PID 4100)..." >> $LOG
wait 4100 2>/dev/null || true
echo "DLBCL training done: $(date)" >> $LOG

# Run BreaKHis no-aug ablation (#017)
echo "Starting BreaKHis no-aug ablation..." >> $LOG
conda run -n ai_dt python src/training/train.py --config configs/breakhis_no_aug.yaml >> $LOG 2>&1
echo "Ablation done: $(date)" >> $LOG

# Run DLBCL evaluation (#015)
echo "Starting DLBCL evaluation..." >> $LOG
conda run -n ai_dt python src/training/evaluate.py --config configs/dlbcl.yaml --checkpoint results/checkpoints/best_dlbcl.pth >> $LOG 2>&1
echo "DLBCL eval done: $(date)" >> $LOG

echo "=== Pipeline complete: $(date) ===" >> $LOG
