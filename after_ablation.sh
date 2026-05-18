#!/bin/bash
PROJECT=/mnt/h/1.MasterDegree/17.RD/project
LOG=$PROJECT/pipeline.log
cd $PROJECT

echo "=== Waiting for ablation (PID 5037) ===" >> $LOG
until ! kill -0 5037 2>/dev/null; do sleep 10; done
echo "Ablation done: $(date)" >> $LOG

# Start DLBCL training from scratch (#013)
echo "Starting DLBCL training (#013) from scratch..." >> $LOG
conda run -n ai_dt python src/training/train.py --config configs/dlbcl.yaml >> $LOG 2>&1
echo "DLBCL training done: $(date)" >> $LOG

# Run DLBCL evaluation (#015)
echo "Starting DLBCL evaluation (#015)..." >> $LOG
conda run -n ai_dt python src/training/evaluate.py \
    --config configs/dlbcl.yaml \
    --checkpoint results/checkpoints/best_dlbcl.pth >> $LOG 2>&1
echo "DLBCL eval done: $(date)" >> $LOG
