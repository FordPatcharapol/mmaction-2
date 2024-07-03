from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'E:/Ford/CP_Match/prototype/mmaction2/work_dirs/tsn_custom/tsn_custom.py'
checkpoint_file = 'E:/Ford/CP_Match/prototype/mmaction2/work_dirs/tsn_custom/epoch_100.pth'
video_file = './data/lotus/val/shoplifter7.mp4'
label_file = './custom_label_map.txt'

# init model
model = init_recognizer(config_file, checkpoint_file,
                        device='cuda:0')  # or device='cuda:0'

# inferance
pred_result = inference_recognizer(model, video_file)
pred_scores = pred_result.pred_score.tolist()
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
top_label = score_sorted[:2]

labels = open(label_file).readlines()
labels = [x.strip() for x in labels]

results = [(labels[k[0]], k[1]) for k in top_label]

print('The top labels with corresponding scores are:')
print(results)
