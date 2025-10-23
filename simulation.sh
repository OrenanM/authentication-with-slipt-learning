#!/bin/bash
# ================================================================
# Script: run_sims_lr.sh
# Descrição:
#   Varre diferentes learning rates, nomeando as execuções como
#   "simulation 1", "simulation 2", ...
#   Requer o fedsplit_runner.py com suporte a --run-name.
# ================================================================

# Caminho do runner (ajuste se necessário)
RUNNER="fedsplit_runner.py"

# Lista de learning rates a testar (edite à vontade)
LEARNING_RATES=(1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6)

# Parâmetros padrão (ajuste se quiser)
NUM_CLIENTS=53
ROUNDS=150
LOCAL_EPOCHS=1
BATCH_SIZE=8
SEED=42
SPLIT="--split"   # remova esta flag se quiser FedAvg "não-split"
DEVICE="auto"

mkdir -p results

echo "=== Iniciando varredura de Learning Rate ==="
i=1
for LR in "${LEARNING_RATES[@]}"; do
  RUN_NAME="simulation ${i}"
  echo ""
  echo ">>> Rodando ${RUN_NAME} (lr=${LR}) ..."
  echo "-----------------------------------------------"

  python "$RUNNER" \
    $SPLIT \
    --num-clients $NUM_CLIENTS \
    --rounds $ROUNDS \
    --local-epochs $LOCAL_EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --device $DEVICE \
    --seed $SEED \
    --run-name "$RUN_NAME"

  echo "Concluído: ${RUN_NAME}"
  echo "-----------------------------------------------"
  ((i+=1))
done

echo "=== Todas as simulações finalizadas ==="
