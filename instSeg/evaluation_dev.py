import numpy as np
from scipy.spatial import distance_matrix
from skimage.measure import regionprops
from scipy.spatial import distance_matrix

'''
Author: Long Chen
Support:
    - higher dimensional data
    - evaluation in 'area' mode and 'curve' mode
    - input as label map or stacked binary maps
    - matrics: 
        - averagePrecision, aggregatedPricision
        - averageRecall, aggregatedRecall
        - averageF1, aggregatedF1
        - aggregatedJaccard, instanceAveragedJaccard
        - aggregatedDice, instanceAveragedDice
        - SBD (symmetric best Dice)
'''

def map2stack(M, bg_label=0):
    '''
    Args:
        M: H x W x (1)
        bg_label: label of the background
    Return:
        S: C x H x W
    '''
    M = np.squeeze(M)
    labels = np.unique(M[M!=bg_label])
    S = np.ones((len(labels), M.shape[0], M.shape[1]), bool)
    for idx, l in enumerate(labels):
        if l == bg_label:
            continue
        S[idx] = (M==l)
    return S

class Sample(object):

    """
    class for evaluating a singe prediction-gt pair
    """

    def __init__(self, pd, gt, dimension=2, mode='area', tolerance=3, allow_overlap=False):

        '''
        Args:
            pd: numpy array of dimension D or D+1/list of dimension D
            gt: numpy array of dimension D or D+1/list of dimension D
            dimension: dimension D of the image / ground truth
            mode: 'area' / 'centroid' / 'curve', evaluate area / centroid indicated position / curve
            tolerance: int, shift tolerance, only valid when mode='centroid' / 'curve'
        Note:
            D + 1 is not supported in 'centroid' mode
            pd/gt can be giveb by:
                - a label map of dimension D, with 0 indicating the background
                - a binary map of demension (D+1) with each instance occupying one channel of the first dimension
            The binary map costs more memory, but can handle overlapped object. If objects are not overlapped, use the label map to save memory and accelarate the computation.
        '''
        self.ndim = dimension
        self.mode = mode
        self.tolerance = tolerance
        self.allow_overlap = allow_overlap

        if isinstance(pd, list):
            pd = np.array(pd) if len(pd) != 0 else np.zeros((0,10,10))
        if isinstance(gt, list):
            gt = np.array(gt) if len(gt) != 0 else np.zeros((0,10,10))

        assert (gt.ndim == dimension) or (gt.ndim == dimension+1) or gt.shape[0] == 0
        assert (pd.ndim == dimension) or (pd.ndim == dimension+1) or pd.shape[0] == 0

        if pd.ndim == dimension:
            pd = map2stack(pd)
        if gt.ndim == dimension:
            gt = map2stack(gt)

        self.gt, self.pd = gt > 0, pd > 0
        
        # remove 'empty' object in gt, and save size of all objects in gt
        self.S_gt = np.sum(self.gt, axis=tuple(range(1, 1+dimension)))
        self.gt = self.gt[self.S_gt > 0]
        self.S_gt = self.S_gt[self.S_gt>0]
        # self.S_gt = {l: c for l, c in enumerate(self.S_gt[self.S_gt>0])}
        
        # remove 'empty' object in predcition, and save size of all objects in prediction
        self.S_pd = np.sum(self.pd, axis=tuple(range(1, 1+dimension)))
        self.pd = self.pd[self.S_pd > 0]
        self.S_pd = self.S_pd[self.S_pd>0]
        # self.S_pd = {l: c for l, c in enumerate(self.S_pd[self.S_pd>0])}

        self.N_gt, self.N_pd = len(self.S_gt), len(self.S_pd)
        self.intersection = None
        self.jaccard = None
        self.dice = None

        self.matches = {}

        # the max-overlap match is not symmetric, thus, store them separately
        # self.match_pd = None  # (prediction label)-(matched gt label)
        # self.intersection_pd = None # (prediction label)-(intersection area)
        # self.match_gt = None # (gt label)-(matched prediction label)
        # self.intersection_gt = None # (gt label)-(intersection area)

        # # precision 
        # self.precision_pd, self.precision_gt = None, None
        # # recall
        # self.recall_pd, self.recall_gt = None, None
        # # F1 score
        # self.f1_pd, self.f1_gt = None, None
        # # dice
        # self.dice_pd, self.dice_gt = None, None
        # # jaccard
        # self.jaccard_pd, self.jaccard_gt = None, None

        # # aggreated area
        # self.agg_intersection = None
        # self.agg_union = None
        # self.agg_area = None
        # # match count, computed with respect to ground truth (which makes sense)
        # self.gt_match_count = {}
        # self.pd_match_count = {}
    

    def _intersection(self):
        '''
        compute the intersection between prediction and ground truth
        Return:
            match: dict of the best match
            intersection: dict of the intersection area
        '''
        
        if self.intersection is not None:
            return self.intersection
        
        if self.mode == "area":
            self.intersection = np.zeros((self.N_pd, self.N_gt))
            for idx in range(self.N_pd):
                overlap = np.sum(np.multiply(self.gt, np.expand_dims(self.pd[idx], axis=0)), axis=tuple(range(1, 1+self.ndim)))
                self.intersection[idx] = overlap
        
        elif self.mode == "centroid":
            pass

        self.dice = self.intersection * 2 / (np.expand_dims(self.S_pd, axis=1) + np.expand_dims(self.S_gt, axis=0))
        self.jaccard = self.intersection / (np.expand_dims(self.S_pd, axis=1) + np.expand_dims(self.S_gt, axis=0) - self.intersection)

        return self.intersection
    
    def _match(self, thres):
        '''
        Args:
            thres: threshold to determine the a match
            metric: metric used to determine match, 'Jaccard' or 'Dice'
        Retrun:
            match_count, gt_count: the number of matches, the number of matched gt objects
        '''
        self._intersection()
        if thres is None:
            return None
        if thres not in self.matches.keys():
            if self.allow_overlap or thres < 0.5:
                pass
            else:
                self.matches[thres] = self.jaccard > thres
        assert np.sum(self.matches[thres]) == np.count_nonzero(self.matches[thres])
        return self.matches[thres]

    # def _getSegPrecision(self, subject='pred'):

    #     '''
    #     compute the segmentation precision of each 'subject' object
    #     '''
    #     if subject == 'pred' and self.precision_pd is None:    
    #         self._intersection('pred')
    #         self.precision_pd = {k: self.intersection_pd[k] / self.S_pd[k] for k in self.match_pd.keys()}

    #     if subject == 'gt' and self.precision_gt is None:    
    #         self._intersection('gt')
    #         self.precision_gt = {k: self.intersection_gt[k] / self.S_gt[k] for k in self.match_gt.keys()}


    # def _getSegRecall(self, subject='pred'):

    #     '''
    #     compute the segmentation recall of each 'subject' object
    #     '''
    #     if subject == 'pred' and self.recall_pd is None:    
    #         self._intersection('pred')
    #         self.recall_pd = {}
    #         for k, m in self.match_pd.items():
    #             self.recall_pd[k] = self.intersection_pd[k] / self.S_gt[m] if m is not None else 0

    #     if subject == 'gt' and self.recall_gt is None:    
    #         self._intersection('gt')
    #         self.recall_gt = {}
    #         for k, m in self.match_gt.items():
    #             self.recall_gt[k] = self.intersection_gt[k] / self.S_pd[m] if m is not None else 0


    # def _getSegF1(self, subject='pred'):
    #     '''
    #     compute the segmentation f1 score of each 'subject' object
    #     '''
    #     self._getSegPrecision(subject)
    #     self._getSegRecall(subject)

    #     if subject == 'pred' and self.f1_pd is None:
    #         self.f1_pd = {}
    #         for k, p in self.precision_pd.items():
    #             self.f1_pd[k] = 2*(p*self.recall_pd[k])/(p + self.recall_pd[k] + 1e-8)

    #     if subject == 'gt' and self.f1_gt is None:
    #         self.f1_gt = {}
    #         for k, p in self.precision_gt.items():
    #             self.f1_gt[k] = 2*(p*self.recall_gt[k])/(p + self.recall_gt[k] + 1e-8)
    

    # def _getSegJaccard(self, subject='pred'):

    #     '''
    #     compute the segmentation Jaccard index of each 'subject' object
    #     '''
    #     self._intersection(subject)
        
    #     if subject == 'pred' and self.jaccard_pd is None:
    #         match, intersection = self.match_pd, self.intersection_pd
    #         area_sub, area_ref = self.S_pd, self.S_gt
    #     elif subject == 'gt' and self.jaccard_gt is None:
    #         match, intersection = self.match_gt, self.intersection_gt
    #         area_sub, area_ref = self.S_gt, self.S_pd
    #     else:
    #         return None

    #     jaccard = {}
    #     for k, m in match.items():
    #         union = area_sub[k] - intersection[k]
    #         if m is not None:
    #             union += area_ref[m]
    #         jaccard[k] = intersection[k] / union
        
    #     if subject == 'pred':
    #         self.jaccard_pd = jaccard
    #     else:
    #         self.jaccard_gt = jaccard


    # def _getSegDice(self, subject='pred'):
        
    #     '''
    #     compute the segmentation Dice index of each 'subject' object
    #     '''
    #     self._intersection(subject)

    #     if subject == 'pred' and self.dice_pd is None:
    #         match, intersection = self.match_pd, self.intersection_pd
    #         area_sub, area_ref = self.S_pd, self.S_gt
    #     elif subject == 'gt' and self.dice_gt is None:
    #         match, intersection = self.match_gt, self.intersection_gt
    #         area_sub, area_ref = self.S_gt, self.S_pd
    #     else:
    #         return None

    #     dice = {}
    #     for k, m in match.items():
    #         agg_area = area_sub[k] + area_ref[m] if m is not None else area_sub[k]
    #         dice[k] = 2 * intersection[k] / agg_area
        
    #     if subject == 'pred':
    #         self.dice_pd = dice
    #     else:
    #         self.dice_gt = dice
        

    # def averageSegPrecision(self, subject='pred'):

    #     '''
    #     average of the segmentation precision of each 'subject' object
    #     '''
    #     if self.mode == 'centroid':
    #         raise Exception("averageSegPrecision is not a valid score in 'centroid' mode")

    #     self._getSegPrecision(subject)
    #     if subject == 'pred':
    #         return np.mean(list(self.precision_pd.values()))
    #     else:
    #         return np.mean(list(self.precision_gt.values()))


    # def averageSegRecall(self, subject='pred'):

    #     '''
    #     average of the segmentation recall of each 'subject' object
    #     '''
    #     if self.mode == 'centroid':
    #         raise Exception("averageSegRecall is not a valid score in 'centroid' mode")
        
    #     self._getSegRecall(subject)
    #     if subject == 'pred':
    #         return np.mean(list(self.recall_pd.values()))
    #     else:
    #         return np.mean(list(self.recall_gt.values())) 
             

    # def averageSegF1(self, subject='pred'):
    #     '''
    #     average of the segmentation F1 score of each 'subject' object
    #     '''
    #     if self.mode == 'centroid':
    #         raise Exception("averageSegF1 is not a valid score in 'centroid' mode")
        
    #     self._getSegF1(subject)
    #     if subject == 'pred':
    #         return np.mean(list(self.f1_pd.values()))
    #     else:
    #         return np.mean(list(self.f1_gt.values()))
        
    def averageDice(self, thres=None, subject='pd'):
        max_axis = 1 if subject == 'pd' else 0
        return np.mean(np.amax(self.dice, axis=max_axis))

    def averageJaccard(self, thres=None, subject='pd'):
        max_axis = 1 if subject == 'pd' else 0
        return np.mean(np.amax(self.jaccard, axis=max_axis))

    def aggregatedJaccard(self):
        '''  
        Reference:
            A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology
        '''
        self._intersection()
        idx = np.argmax(self.intersection, axis=0)

        C = np.sum(self.intersection[idx, list(range(self.N_gt))])
        U = np.sum(self.S_gt) + np.sum(self.S_pd[idx]) - C + np.sum(self.S_pd[list(set(range(self.N_pd))-set(idx))])
        print(np.sum(self.S_gt) + np.sum(self.S_pd[idx]) - C)
        print(np.sum(self.S_gt), np.sum(self.S_pd[idx]))
        print(list(set(range(self.N_pd))-set(idx)))
        print(np.sum(self.S_pd[list(set(range(self.N_pd))-set(idx))]))

        print(idx, C, U)

        return 1 if (C == 0 and U == 0) else C/U
    
    def AJI(self): # alias of aggregatedJaccard (aggregated Jaccard index)
        return self.aggregatedJaccard()


    def SBD(self):
        '''
        symmetric best dice
        '''
        return min(self.averagedDice(subject='pd'), self.averagedDice(subject='gt'))


    def detectionRecall(self, thres=0.5):
        self._match(thres=thres)
        if self.N_gt != 0:
            return np.sum(self.matches[thres])/self.N_gt
        else:
            return 1

    def detectionPrecision(self, thres=0.5):
        self._match(thres=thres)
        if self.N_pd != 0:
            return np.sum(self.matches[thres])/self.N_pd
        else:
            return 1

    # def AP_COCO(self, thres=None, metric='Jaccard', interpolated=True):
    #     '''
    #     average precision based on MS COCO definition: https://cocodataset.org/#home
    #     in case of objects of the same class, AP == mAP in COCO definition
    #     '''
    #     if self.mode == 'centroid':
    #         raise Exception("AP_COCO does not make sense in the 'centroid' mode")
    #     else:
    #         thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] if thres is None else thres
    #         dps = [self.detectionPrecision(thres=t, metric=metric) for t in thres]
    #         if interpolated:
    #             dps = [max(dps[i:-1]) for i in range(len(dps)-1)] + [dps[-1]]
    #         return np.mean(dps)

    # def AFNR(self, thres=None, metric='Jaccard'):
    #     '''
    #     average false-negative ratio, ref.:
    #         Edlund, C., Jackson, T.R., Khalid, N. et al. LIVECell—A large-scale dataset for label-free live cell segmentation. Nat Methods (2021). https://doi.org/10.1038/s41592-021-01249-6
    #     '''
    #     if self.mode == 'centroid':
    #         raise Exception("AP_COCO does not make sense in the 'centroid' mode")
    #     else:
    #         thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] if thres is None else thres
    #         fnr = [1 - self.detectionRecall(thres=t, metric=metric) for t in thres]
    #         return np.mean(fnr)

    def P_DSB(self, thres=0.5):
        '''
        the precision based on Data Scient Bowl 2018 definition: https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''
        self._match(thres=thres)
        N_inter = np.sum(self.matches[thres])
        N_union = self.N_pd + self.N_gt - N_inter
        return N_inter/N_union if N_union != 0 else 1

    def AP_DSB(self, thres=None):
        '''
        average precision based on Data Scient Bowl 2018 definition: https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''
        if self.mode == 'centroid':
            return self.P_DSB()
        else:
            thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] if thres is None else thres
            ps = [self.P_DSB(thres=t) for t in thres]
            return np.mean(ps)
        
    def RQ(self, thres=0.5):
        '''
        recognition quality
        defined in paper "Panoptic Segmentation": https://arxiv.org/abs/1801.00868
        '''
        self._match(thres=thres)
        N_inter = np.sum(self.matches[thres])
        N_union = self.N_pd + self.N_gt
        return 2*N_inter/N_union if N_union != 0 else 1

    def SQ(self):
        '''
        segmentation quality
        defined in paper "Panoptic Segmentation": https://arxiv.org/abs/1801.00868
        '''
        thres = 0.5
        self._match(thres=thres)
        rr, cc = np.nonzero(self.matches[thres])
        return np.mean(self.jaccard[rr, cc])

    def PQ(self):
        return self.SQ() * self.RQ(thres=0.5)


class GFG(object):   
    # maximal Bipartite matching. 
    def __init__(self, graph): 
          
        self.graph = graph
        # number of applicants  
        self.ppl = len(graph)
        # number of jobs 
        self.jobs = len(graph[0]) 
  
    # A DFS based recursive function 
    # that returns true if a matching  
    # for vertex u is possible 
    def bpm(self, u, matchR, seen): 
        for v in range(self.jobs): 
            # If applicant u is interested in job v and v is not seen 
            if self.graph[u][v] and seen[v] == False: 
                seen[v] = True 
                '''If job 'v' is not assigned to 
                   an applicant OR previously assigned  
                   applicant for job v (which is matchR[v])  
                   has an alternate job available.  
                   Since v is marked as visited in the  
                   above line, matchR[v]  in the following 
                   recursive call will not get job 'v' again'''
                if matchR[v] == -1 or self.bpm(matchR[v], matchR, seen): 
                    matchR[v] = u 
                    return True, v
        return False
    
    def maxBPM(self): 
        ''' returns maximum number of matching ''' 
        # applicant number assigned to job i, the value -1 indicates nobody is assigned
        matchR = [-1] * self.jobs   
        # Count of jobs assigned to applicants 
        result = 0 
        for i in range(self.ppl): 
            # Mark all jobs as not seen for next applicant. 
            seen = [False] * self.jobs 
            # Find if the applicant 'u' can get a job 
            if self.bpm(i, matchR, seen): 
                result += 1
        return result, matchR 

class Evaluator(object):


    def __init__(self, dimension=2, mode='area', tolerance=3, verbose=True):

        self.ndim = dimension
        self.mode = mode
        self.tolerance = tolerance

        self.examples = []
        self.total_pd = 0
        self.total_gt = 0

        self.verbose = verbose
        

    def add_example(self, pred, gt):
        e = Sample(pred, gt, dimension=self.ndim, mode=self.mode, tolerance=self.tolerance)
        self.examples.append(e)
        self.total_pd += e.N_pd
        self.total_gt += e.N_gt
        if self.verbose:
            print("example added, total: ", len(self.examples))


    def meanAggregatedJaccard(self):
        '''
        aggregatedJaccard: accumulate area over images first, then compute the AJI
        meanAggregatedJaccard: compute AJI of each image, and then take the average
        '''
        if self.mode == 'centroid':
            raise Exception("mAJ is not a valid score in 'centroid' mode")

        AJs = [e.aggregatedJaccard() for e in self.examples]
        mAJ = np.mean(AJs)
        if self.verbose:
            print('mAJ (mean aggregated Jaccard): ', mAJ)
        return mAJ    
    
    def mAJI(self):
        return self.meanAggregatedJaccard()

    def aggregatedJaccard(self):
        ''' 
        aggregatedJaccard: accumulate area over images first, then compute the AJI
        meanAggregatedJaccard: compute AJI of each image, and then take the average
        Reference:
            A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology
        '''
        if self.mode == 'centroid':
            raise Exception("aggregatedJaccard is not a valid score in 'centroid' mode")

        agg_intersection, agg_union = 0, 0
        for e in self.examples:
            agg_i, agg_u, _ = e.accumulate_area()
            agg_intersection += agg_i
            agg_union += agg_u
        if self.verbose:
            print('aggregated Jaccard: ', agg_intersection/agg_union)

        return agg_intersection/agg_union

    def AJI(self):
        return self.aggregatedJaccard()
    
    def meanAggregatedDice(self):
        '''
        aggregatedDice: accumulate area over images first, then compute the ADS
        meanAggregatedDice: compute ADS of each image, and then take the average
        '''
        if self.mode == 'centroid':
            raise Exception("mAD is not a valid score in 'centroid' mode")

        ADs = [e.aggregatedDice() for e in self.examples]
        mAD = np.mean(ADs)
        print('mAD (mean aggregated Dice): ', mAD)
        return mAD 

    def mADS(self):
        return self.meanAggregatedDice()

    def aggregatedDice(self):
        ''' 
        no defination found, derived from aggregated Jaccard Index
        Reference:
            CNN-BASED PREPROCESSING TO OPTIMIZE WATERSHED-BASED CELL SEGMENTATION IN 3D CONFOCAL MICROSCOPY IMAGES
        aggregatedDice: accumulate area over images first, then compute the ADS
        meanAggregatedDice: compute ADS of each image, and then take the average
        '''
        if self.mode == 'centroid':
            raise Exception("aggregatedDice is not a valid score in 'centroid' mode")

        agg_intersection, agg_area = 0, 0
        for e in self.examples:
            agg_i, _, agg_a = e.accumulate_area()
            agg_intersection += agg_i
            agg_area += agg_a
        if self.verbose:
            print('aggregated Dice: ', 2*agg_intersection/agg_area)

        return 2*agg_intersection/agg_area

    def ADS(self):
        return self.aggregatedDice()


    def mSBD(self):
        '''
        mean of SBD (symmetric best dice) of each image
        '''

        if self.mode == 'centroid':
            raise Exception("mSBD is not a valid score in 'centroid' mode")

        SBDs = [e.SBD() for e in self.examples]
        mSBD = np.mean(SBDs)
        print('mSBD (mean symmetric best dice): ', mSBD)
        return mSBD 

    

    def detectionRecall(self, thres=0.5, metric='Jaccard'):

        match_count, N_gt = 0, 0
        for e in self.examples:
            pd_match_gt_count = e.match(thres=thres, metric=metric, subject='pred')[1]
            match_count += pd_match_gt_count
            N_gt += e.N_gt
        S = match_count/N_gt if N_gt > 0 else 1
        if self.verbose:
            print("detectionRecall over the whole dataset under '" + metric + "' {}: {}".format(thres, S))
        return S

    def detectionPrecision(self, thres=0.5, metric='Jaccard'):

        match_count, N_pd = 0, 0
        for e in self.examples:
            pd_match_pd_count = e.match(thres=thres, metric=metric, subject='pred')[0]
            match_count += pd_match_pd_count
            N_pd += e.N_pd
        A = match_count/N_pd if N_pd > 0 else 1
        if self.verbose:
            print("detectionPrecision over the whole dataset under '" + metric + "' {}: {}".format(thres, A))
        return A


    def P_DSB(self, thres=0.5, metric='Jaccard'):
        '''
        the precision based on Data Scient Bowl 2018 definition: https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''

        match_count_gt, match_count_pd, N_pd, N_gt = 0, 0, 0, 0
        for e in self.examples:
            match_count = e.match(thres=thres, metric=metric, subject='gt')
            match_count_gt += match_count[0]
            match_count_pd += match_count[1]
            N_pd += e.N_pd
            N_gt += e.N_gt
            # print('===================')
            # print(match_count[0], match_count[1], e.N_pd, e.N_gt)
            # print(match_count_gt, match_count_pd, N_pd, N_gt)
        union = N_gt + N_pd - match_count_pd
        # print('union', union)
        # it is possible that gt, pred are both empty
        P = match_count_gt/union if union > 0 else 1
        if self.verbose:
            print("P (Data Scient Bowl 2018) over the whole dataset under '" + metric + "' {}: {}".format(thres, P))
        return P

    def AP_DSB(self, thres=None, metric='Jaccard', interpolated=False):
        '''
        average precision based on Data Scient Bowl 2018 definition: https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''
        if self.mode == 'centroid':
            AP = self.P_DSB()
        else:
            thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] if thres is None else thres
            ps = [self.P_DSB(thres=t, metric=metric) for t in thres]
            if interpolated:
                ps = [max(ps[i:-1]) for i in range(len(ps)-1)] + [ps[-1]]
            AP = np.mean(ps)
        if self.verbose:
            print('AP (Data Scient Bowl 2018) over the whole dataset: ', AP)
        return AP


    def AP_COCO(self, thres=None, metric='Jaccard', interpolated=True):
        '''
        average precision based on MS COCO definition: https://cocodataset.org/#home
        in case of objects of the same class, AP == mAP in COCO definition
        '''
        if self.mode == 'centroid':
            raise Exception("AP_COCO does not make sense in the 'centroid' mode")
        else:
            thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] if thres is None else thres
            dps = [self.detectionPrecision(thres=t, metric=metric) for t in thres]
            if interpolated:
                dps = [max(dps[i:-1]) for i in range(len(dps)-1)] + [dps[-1]]
            AP = np.mean(dps)
        if self.verbose:
            print('AP (MS COCO) over the whole dataset: ', AP)
        return AP


    def AFNR(self, thres=None, metric='Jaccard'):
        '''
        average false-negative ratio, ref.:
            Edlund, C., Jackson, T.R., Khalid, N. et al. LIVECell—A large-scale dataset for label-free live cell segmentation. Nat Methods (2021). https://doi.org/10.1038/s41592-021-01249-6
        '''
        if self.mode == 'centroid':
            raise Exception("AP_COCO does not make sense in the 'centroid' mode")
        else:
            thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] if thres is None else thres
            fnr = [1 - self.detectionRecall(thres=t, metric=metric) for t in thres]
            AFNR = np.mean(fnr)
        if self.verbose:
            print('AFNR (average false-negative ratio) over the whole dataset: ', AFNR)
        return AFNR



class Evaluator_Seg(object):

    def __init__(self, tolerance=0, verbose=True):
        self.T = tolerance
        self.verbose = verbose
        self.tp = 0
        self.fp = 0
        self.fn = 0


    def add_example(self, pred, gt):
        pred, gt = pred > 0, gt > 0
        tp = np.logical_and(pred, gt)
        fp = np.logical_xor(tp, pred)
        fn = np.logical_xor(tp, gt)

        if self.T > 0 and np.sum(gt) > 0:
            
            cood_gt = np.array(np.nonzero(gt)).T
            cood_fp = np.array(np.nonzero(fp)).T
            if len(cood_fp) != 0 and len(cood_gt) != 0:
                D = distance_matrix(cood_fp, cood_gt).min(axis=1)
                Nfp = np.sum(D > self.T)
            else:
                Nfp = len(cood_fp)
            
            cood_pred = np.array(np.nonzero(pred)).T
            cood_fn = np.array(np.nonzero(fn)).T
            if len(cood_fn) != 0 and len(cood_pred) != 0:
                D = distance_matrix(cood_fn, cood_pred).min(axis=1)
                Nfn = np.sum(D > self.T)
            else:
                Nfn = len(cood_fn)
        else:
            Nfp, Nfn = np.sum(fp), np.sum(fn)
        Ntp = np.sum(tp)

        self.tp += Ntp
        self.fp += Nfp
        self.fn += Nfn
    
    def JI(self):
        ji = self.tp / (self.tp + self.fp + self.fn)
        if self.verbose:
            print("Jaccard Index: ", ji)
        return ji
    
    def precision(self):
        pr = self.tp / (self.tp + self.fp)
        if self.verbose:
            print("Precision: ", pr)
        return pr
        
    def recall(self):
        recall = self.tp / (self.tp + self.fn)
        if self.verbose:
            print("Recall: ", recall)
        return recall


if __name__ == '__main__':
    from skimage.io import imread
    import time
    import numpy as np
    import glob

    #### test1 ####

    gt = np.zeros((64, 64*6), np.uint8)
    pred = np.zeros((64, 64*6), np.uint8)

    P = 64
    D = 20

    for i in range(6):
        gt[P//2-D:P//2+D, P//2+P*i-D:P//2+P*i+D] = i + 1
    
    # perfect
    pred[P//2-D:P//2+D, P//2-D:P//2+D] = 1
    # miss aligned
    pred[P//2-D-5:P//2+D-5, P//2+P*1-D-5:P//2+P*1+D-5] = 2
    # under-segmentation
    pred[P//2-D+5:P//2+D-5, P//2+P*2-D+5:P*2+P//2+D-5] = 3
    # over-segmentation
    pred[P//2-D-5:P//2+D+5, P//2+P*3-D-5:P//2+P*3+D+5] = 4
    # miss
    pred[P//2-D:P//2+D, P//2+P*4-D:P//2+P*4+D] = 0
    # split
    pred[P//2-D:P//2+D, P//2+P*5-D:P//2+P*5] = 5
    pred[P//2-D:P//2+D, P//2+P*5:P//2+P*5+D] = 6


    #### toy test ####
    # gt = imread('./test/toy_example/gt.png')
    # pred = imread('./test/toy_example/pred.png')
    
    sample = Sample(pred, gt, mode='area')

    sub = 'pd'
    thres = 0.5
    sample._intersection()
    print(sample.intersection)
    # print(sample.dice)
    print(sample.jaccard)
    m = sample._match(0.6)
    print('averageJaccard', sample.averageJaccard(subject=sub))
    print('averageDice', sample.averageDice(subject=sub))
    print('aggregatedJaccard', sample.aggregatedJaccard())
    print('detection recall', sample.detectionRecall(thres))
    print('detection precision', sample.detectionPrecision(thres))


    # print('COCO AP', sample.AP_COCO())
    # print('average false negative ratio', sample.AFNR())
    print('DSB P', sample.P_DSB(thres=0.5))
    print('DSB P', sample.P_DSB(thres=0.6))
    print('DSB P', sample.P_DSB(thres=0.7))
    print('DSB AP', sample.AP_DSB())
    print('RQ', sample.RQ())
    print('SQ', sample.SQ())
    print('PQ', sample.PQ())



    # f_gts = sorted(glob.glob('./test/cell/gt/*.png'))[0:50]
    # f_preds = sorted(glob.glob('./test/cell/pred/*.tif'))[0:50]

    # # evalation of a whole dataset
    # e = Evaluator(dimension=2, mode='area')
    # for f_gt, f_pred in zip(f_gts, f_preds):
    #     pred = imread(f_pred)
    #     gt = imread(f_gt)
    #     # add one segmentation
    #     e.add_example(pred, gt)

    # e.detectionRecall()
    # e.detectionPrecision()
    # e.AP_COCO()
    # e.AP_DSB()
    # e.AFNR()
    # e.AJI()
    # e.ADS()