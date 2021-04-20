import numpy as np
import os
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from scipy import stats



def get_cxcy_calibrationplot(cxcy, dtId, dtScores, tps, dtIg):
    cxcy[cxcy>1] = 1
    binx = np.linspace(0,1,11)
    ret_xy = stats.binned_statistic(cxcy, None, 'count', bins=[binx])
    bin_idx = [ [] for _ in range(10) ]

    print(max(ret_xy.binnumber))

    for i in range(len(dtId)):
        bin_idx[ret_xy.binnumber[i]-1].append(i)

    cxcy_result = np.zeros((5,10))

    for i in range(10):
        result = getECE(bin_idx[i],dtScores, tps, dtIg)
        cxcy_result[:,i] = result

    plt.plot(np.linspace(0.05,0.95,10),cxcy_result[0,:],marker='o',label='iou_{}'.format(0.5))
    #plt.plot(np.linspace(0.05,0.95,10),cx_result[1,:],marker='o',label='iou_{}'.format(0.6))
    #plt.plot(np.linspace(0.05,0.95,10),cx_result[2,:],marker='o',label='iou_{}'.format(0.7))
    #plt.plot(np.linspace(0.05,0.95,10),cx_result[3,:],marker='o',label='iou_{}'.format(0.8))
    #plt.plot(np.linspace(0.05,0.95,10),cx_result[4,:],marker='o',label='iou_{}'.format(0.9))
    plt.xlim(0,1)
    plt.legend()
    plt.show()


def getECE(bin_idx, dtScores, tps, dtIg):

    area_dtScores = dtScores[bin_idx]
    inds = np.argsort(-area_dtScores, kind='mergesort')

    area_tps = tps[:,bin_idx]
    area_dtIg = dtIg[:,bin_idx]

    sorted_area_dtScores = area_dtScores[inds]
    sorted_tps = area_tps[:,inds]
    sorted_dtIg = area_dtIg[:,inds]

    CONFIDENCE_SCORE = np.zeros((5,10))
    MATCHED_RESILT = np.zeros((5,10))

    for k in range(10):
        for j in range(5):
            confidence_score = []
            matched_result = []
            for i in range(sorted_area_dtScores.size):
                if(sorted_area_dtScores[i]>0.1*k and sorted_area_dtScores[i]<0.1*(k+1) and sorted_dtIg[j][i]==False):
                    confidence_score.append(sorted_area_dtScores[i])
                    matched_result.append(sorted_tps[j][i])
                    if i == len(sorted_area_dtScores)-1:
                        continue
                    if (sorted_area_dtScores[i+1]<0.1*k):
                        continue
            CONFIDENCE_SCORE[j,k] = np.mean(confidence_score)
            MATCHED_RESILT[j,k] =  np.mean(matched_result)

    counts, edge = np.histogram(area_dtScores, bins=np.linspace(0,1,11))
    mm = np.abs(CONFIDENCE_SCORE-MATCHED_RESILT) * (counts/sum(counts))
    return np.sum(mm,axis=1)


annot_path = os.path.join('F:\TUM Learning Material\Forschung\CenterNet\CenterNet-master\data\coco', 'annotations','instances_{}2017.json').format('val')
cocoGt = coco.COCO(annot_path)
cocoDt = cocoGt.loadRes('{}/centernet_hourglass_results.json'.format('F:\TUM Learning Material\Forschung\YunmingBai_EvaluationScript\DetectionResults'))
#cocoDt = cocoGt.loadRes('F:\TUM Learning Material\Forschung\YunmingBai_EvaluationScript\DetectionResults\maskrcnn_detection_results.json')
#cocoDt = cocoGt.loadRes('F:\TUM Learning Material\Forschung\YunmingBai_EvaluationScript\DetectionResults\yolov5x_predictions.json')
#cocoDt = cocoGt.loadRes('F:\TUM Learning Material\Forschung\YunmingBai_EvaluationScript\DetectionResults\detr_detection_results.json')


coco_eval = COCOeval(cocoGt,cocoDt, "bbox")
coco_eval.evaluate()
coco_evalImgs =  coco_eval.getevalImgs()

E=[]
print("collection_allranges")
for i in range(len(coco_evalImgs)):
    if coco_evalImgs[i]!= None and coco_evalImgs[i]['aRng'] == [0 , 1e5 ** 2]:
        E.append(coco_evalImgs[i])

print("finish collecting areaRange")

maxDet = 100

dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)
dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)
tps = np.logical_and(dtm, np.logical_not(dtIg))

dtId = []

for i in range(len(E)):
    #print(i)
    dtId.extend(E[i]['dtIds'][0:maxDet])

cxcy_alltps = np.zeros((len(dtId),2))

for i in range(len(dtId)):
    loaded_image = cocoGt.loadImgs(cocoDt.loadAnns(dtId[i])[0]['image_id'])
    height = loaded_image[0]['height']
    width = loaded_image[0]['width']
    x, y, w, h = dt_bbox = cocoDt.loadAnns(dtId[i])[0]['bbox']
    cx = (x+0.5*w)/width
    cy = 1 - (y+0.5*h)/height
    cxcy_alltps[i,0] = cx
    cxcy_alltps[i,1] = cy


cx = cxcy_alltps[:,0]
cy = cxcy_alltps[:,1]


get_cxcy_calibrationplot(cx, dtId, dtScores, tps, dtIg)
get_cxcy_calibrationplot(cy, dtId, dtScores, tps, dtIg)



