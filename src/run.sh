# training
python main.py --gru -model ./checkpoints/model.demo > ./logs/log.demo

# testing with ILP
python main.py --analyze --trig_model_path ./checkpoints/model.demo[_timestamp] > ./logs/log.demo_ilp
