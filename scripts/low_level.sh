METHOD='l1'
Crypto='BTCUSDT'
RUN_COUNT=1

# trend
nohup python -u RL/agent/low_level.py --trend_method "$METHOD" --vol_method "$METHOD" --liq_method "$METHOD" --run_count $RUN_COUNT --alpha 1 --clf 'trend' --dataset "$Crypto" --device 'cuda:6' \
    --label trend_1 >./logs/low_level/$Crypto/run_$RUN_COUNT/trend_1_$METHOD.log 2>&1 &
#
nohup python -u RL/agent/low_level.py --trend_method "$METHOD" --vol_method "$METHOD" --liq_method "$METHOD" --run_count $RUN_COUNT --alpha 1 --clf 'trend' --dataset "$Crypto" --device 'cuda:2' \
    --label trend_-1 >./logs/low_level/$Crypto/run_$RUN_COUNT/trend_-1_$METHOD.log 2>&1 &

# vol
nohup python -u RL/agent/low_level.py --trend_method "$METHOD" --vol_method "$METHOD" --liq_method "$METHOD" --run_count $RUN_COUNT --alpha 1 --clf 'vol' --dataset "$Crypto" --device 'cuda:0' \
    --label vol_1 >./logs/low_level/$Crypto/run_$RUN_COUNT/vol_1_$METHOD.log 2>&1 &

nohup python -u RL/agent/low_level.py --trend_method "$METHOD" --vol_method "$METHOD" --liq_method "$METHOD" --run_count $RUN_COUNT --alpha 1 --clf 'vol' --dataset "$Crypto" --device 'cuda:7' \
    --label vol_-1 >./logs/low_level/$Crypto/run_$RUN_COUNT/vol_-1_$METHOD.log 2>&1 &

# liq
nohup python -u RL/agent/low_level.py --trend_method "$METHOD" --vol_method "$METHOD" --liq_method "$METHOD" --run_count $RUN_COUNT --alpha 1 --clf 'liq' --dataset "$Crypto" --device 'cuda:1' \
    --label liq_1 >./logs/low_level/$Crypto/run_$RUN_COUNT/liq_1_$METHOD.log 2>&1 &

nohup python -u RL/agent/low_level.py --trend_method "$METHOD" --vol_method "$METHOD" --liq_method "$METHOD" --run_count $RUN_COUNT --alpha 1 --clf 'liq' --dataset "$Crypto" --device 'cuda:2' \
    --label liq_-1 >./logs/low_level/$Crypto/run_$RUN_COUNT/liq_-1_$METHOD.log 2>&1 &
