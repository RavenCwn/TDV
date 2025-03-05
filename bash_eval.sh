for i in {1..3}; do
    echo "开始第 $i 轮评估..."
    
    # echo "运行 eval_simple_all.py..."
    # python eval_simple_all.py -c result/all_lunch5/02-24_04-05-24  --iters 25 --log_dir log_new1
    
    # echo "运行 eval_different_all.py..."
    # python eval_diff/erent_all.py -c result/longhist/02-26_01-00-58 --iters 25 --log_dir log_long_hist
    python eval_complex_all.py -c results/mlp_gate/02-26_15-42-12 --iters 25 --log_dir log_mlp_gate
    
    echo "运行 eval_complex_all.py..."
    # python eval_complex_all.py -c results/coordiff/stack_blocks/02-24_16-19-56 --iters 25 --log_dir log_stack_only
    # python eval_complex_all.py -c results/mlp_gate/02-26_15-42-12 --iters 25 --log_dir log_mlp_gate
    # python eval_complex_all_sp.py -c result/longhist/02-26_01-00-58 --iters 25 --log_dir log_long_hist
    # python eval_complex_all.py -c result/longhist/02-26_01-00-58 --iters 25 --log_dir log_long_hist
    
    # echo "运行 eval_stack_cup_all.py..."
    # python eval_stack_cup_all.py -c result/all_lunch5/02-24_04-05-24 --iters 25 --log_dir log_new1

    echo "第 $i 轮评估完成"
    echo "-------------------"
done

echo "所有5轮评估已完成"