python3 federated_ser_classifier.py --dataset msp-improv_crema-d --norm znorm \
                        --feature_type emobase --dropout 0.2 --num_epochs 200 --local_epochs 1 \
                        --optimizer adam --model_type fed_avg --learning_rate 0.0005 \
                        --save_dir /media/data/projects/speech-privacy

#chenhj
nohup python3 federated_ser_classifier.py --dataset iemocap --norm znorm \
                        --feature_type emobase --dropout 0.2 --num_epochs 50 --local_epochs 1 \
                        --optimizer adam --model_type fed_avg --learning_rate 0.0005 \
                        --save_dir /home/chenhj/WorkSpaces_python/task001/media/data/projects/speech-privacy > /home/chenhj/WorkSpaces_python/fed-ser-leakage-main/logs/SER_iemocap_50_20240423.log 2>&1 &

nohup python3 federated_ser_classifier.py --dataset msp-improv_crema-d --norm znorm \
                        --feature_type emobase --dropout 0.2 --num_epochs 50 --local_epochs 1 \
                        --optimizer adam --model_type fed_avg --learning_rate 0.0005 \
                        --save_dir /home/chenhj/WorkSpaces_python/task001/media/data/projects/speech-privacy > /home/chenhj/WorkSpaces_python/fed-ser-leakage-main/logs/SER_msp-improv_crema-d_50_20240423.log 2>&1 &


