CRYPTO='BTCUSDT'
METHOD='l1'

nohup python -u preprocess/MAD.py --dataset "$CRYPTO" --method "$METHOD" > ./logs/decomposition/$CRYPTO/run_decomposition_$METHOD.log 2>&1 &

