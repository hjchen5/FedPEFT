taskset 100 python3 train/federated_attribute_attack.py --norm znorm --optimizer adam \
                                    --dataset iemocap --adv_dataset msp-improv_crema-d \
                                    --feature_type emobase --dropout 0.2 --model_type fed_avg \
                                    --learning_rate 0.0005 --local_epochs 1 --num_epochs 200 \
                                    --leak_layer first --model_learning_rate 0.0001 \
                                    --save_dir /media/data/projects/speech-privacy

#chenhj
nohup taskset 100 python3 -u federated_attribute_attack.py --norm znorm --optimizer adam \
                                    --dataset iemocap --adv_dataset msp-improv_crema-d \
                                    --feature_type emobase --dropout 0.2 --model_type fed_avg \
                                    --learning_rate 0.0005 --local_epochs 1 --num_epochs 200 \
                                    --leak_layer first --model_learning_rate 0.0001 \
                                    --save_dir /home/chenhj/WorkSpaces_python/task001/media/data/projects/speech-privacy > /home/chenhj/WorkSpaces_python/fed-ser-leakage-main/logs/attack_iemocap_50_20240423.log 2>&1 &


