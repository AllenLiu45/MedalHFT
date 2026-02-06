Crypto='BTCUSDT'
RUN_COUNT=1

nohup python -u RL/agent/high_level.py --run_count $RUN_COUNT --dataset "$Crypto" --trend_method 'l1' --vol_method 'l1' --liq_method 'l1' --device 'cuda:2' \
    >./logs/high_level/$Crypto/run_$RUN_COUNT/l1_l1_l1.log 2>&1 &

