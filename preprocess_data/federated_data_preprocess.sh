# This is the script for training data preprocess
# audio feature part
# three datasets are iemocap, crema-d, and msp-improv dataset
# cpc, apc, tera, decoar2, audio_albert, distilhubert

taskset 100 python3 preprocess_federate_data.py --dataset msp-improv \
                        --feature_type emobase --norm znorm
                        --data_dir /media/data/sail-data/MSP-IMPROV/MSP-IMPROV 
                        --save_dir /media/data/projects/speech-privacy


#chenhj
taskset 100 python3 preprocess_federate_data.py --dataset msp-improv \
                        --feature_type emobase --norm znorm \
                        --data_dir /home/chenhj/WorkSpaces_python/task001/media/data/sail-data/msp-improv \
                        --save_dir /home/chenhj/WorkSpaces_python/task001/media/data/projects/speech-privacy

taskset 100 python3 preprocess_federate_data.py --dataset iemocap \
                        --feature_type emobase --norm znorm \
                        --data_dir /home/chenhj/WorkSpaces_python/task001/media/data/sail-data/iemocap \
                        --save_dir /home/chenhj/WorkSpaces_python/task001/media/data/projects/speech-privacy

taskset 100 python3 preprocess_federate_data.py --dataset crema-d \
                        --feature_type emobase --norm znorm \
                        --data_dir /home/chenhj/WorkSpaces_python/task001/media/data/sail-data/crema-d \
                        --save_dir /home/chenhj/WorkSpaces_python/task001/media/data/projects/speech-privacy

nohup taskset 100 python3 -u preprocess_federate_data.py --dataset iemocap \
                        --feature_type emobase --norm znorm \
                        --data_dir /home/chenhj/WorkSpaces_python/task001/media/data/sail-data/iemocap \
                        --save_dir /home/chenhj/WorkSpaces_python/task001/media/data/projects/speech-privacy > out.log 2>&1 &