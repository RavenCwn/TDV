for i in {1..5}; do
    echo "开始第 $i 轮评估..."
    
    echo "运行 eval_simple_all.py..."
    python eval_simple_all.py -c results/coordiff/all_lunch2/02-22_20-32-59
    
    echo "运行 eval_different_all.py..."
    python eval_different_all.py -c results/coordiff/all_lunch2/02-22_20-32-59
    
    echo "运行 eval_complex_all.py..."
    python eval_complex_all.py -c results/coordiff/all_lunch2/02-22_20-32-59
    
    echo "第 $i 轮评估完成"
    echo "-------------------"
done

echo "所有5轮评估已完成"