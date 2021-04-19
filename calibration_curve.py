import numpy as np
import os
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt

annot_path = os.path.join('F:\TUM Learning Material\Forschung\CenterNet\CenterNet-master\data\coco', 'annotations','instances_{}2017.json').format('val')
cocoGt = coco.COCO(annot_path)
#cocoDt = cocoGt.loadRes('{}/centernet_hourglass_results.json'.format('F:\TUM Learning Material\Forschung\YunmingBai_EvaluationScript\DetectionResults'))
#cocoDt = cocoGt.loadRes('F:\TUM Learning Material\Forschung\YunmingBai_EvaluationScript\DetectionResults\maskrcnn_detection_results.json')
#cocoDt = cocoGt.loadRes('F:\TUM Learning Material\Forschung\YunmingBai_EvaluationScript\DetectionResults\yolov5x_predictions.json')
cocoDt = cocoGt.loadRes('F:\TUM Learning Material\Forschung\YunmingBai_EvaluationScript\DetectionResults\detr_detection_results.json')


#imgIds = sorted(cocoGt.getImgIds())
#catIds = sorted(cocoGt.getCatIds())

coco_eval = COCOeval(cocoGt,cocoDt, "bbox")
coco_eval.evaluate()
coco_evalImgs =  coco_eval.getevalImgs()

E=[]
print("collection_allranges")
for i in range(len(coco_evalImgs)):
    if coco_evalImgs[i]!= None and coco_evalImgs[i]['aRng'] == [0 , 1e5 ** 2]:
        E.append(coco_evalImgs[i])

print("finish collecting areaRange")

maxDet = 10
# sort all detections evalIngs total area
dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
inds = np.argsort(-dtScores, kind='mergesort')
dtScoresSorted = dtScores[inds]
print(dtScores.shape)
# detection matching matrix
dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
# detection ignore matrix
dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]

# true positive and false positive
# detection but not matched to any ground_truth --> false positive
# detect will be ignored -->  if matched gt ignored / detection out of range
tps = np.logical_and(dtm, np.logical_not(dtIg))


CONFIDENCE_SCORE = np.zeros((5,10))
MATCHED_RESILT = np.zeros((5,10))

print("loop over all iou_thresholds --> minimum iou identify as true positive")
for k in range(10):
    for j in range(5):
        confidence_score = []
        matched_result = []
        for i in range(dtScores.size):
            if(dtScoresSorted[i]>0.1*k and dtScoresSorted[i]<0.1*(k+1) and dtIg[j][i]==False):
                confidence_score.append(dtScoresSorted[i])
                matched_result.append(tps[j][i])
                if i == len(dtScoresSorted)-1:
                    continue
                if (dtScoresSorted[i+1]<0.1*k):
                    continue
        CONFIDENCE_SCORE[j,k] = np.mean(confidence_score)
        MATCHED_RESILT[j,k] =  np.mean(matched_result)

print(CONFIDENCE_SCORE)
print(MATCHED_RESILT)

plt.plot(CONFIDENCE_SCORE[0,:],MATCHED_RESILT[0,:],marker='o',label='iou_{}'.format(0.5))
plt.plot(CONFIDENCE_SCORE[1,:],MATCHED_RESILT[1,:],marker='o',label='iou_{}'.format(0.6))
plt.plot(CONFIDENCE_SCORE[2,:],MATCHED_RESILT[2,:],marker='o',label='iou_{}'.format(0.7))
plt.plot(CONFIDENCE_SCORE[3,:],MATCHED_RESILT[3,:],marker='o',label='iou_{}'.format(0.8))
plt.plot(CONFIDENCE_SCORE[4,:],MATCHED_RESILT[4,:],marker='o',label='iou_{}'.format(0.9))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.legend()
plt.show()





