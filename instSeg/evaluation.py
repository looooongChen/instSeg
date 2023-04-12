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
    S = np.ones((len(labels), M.shape[0], M.shape[1]), np.bool)
    for idx, l in enumerate(labels):
        if l == bg_label:
            continue
        S[idx] = (M==l)
    return S

class Sample(object):

    """
    class for evaluating a singe prediction-gt pair
    """

    def __init__(self, pd, gt, dimension=2, mode='area', tolerance=3):

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
        if isinstance(pd, list):
            pd = np.array(pd) if len(pd) != 0 else np.zeros((0,10,10))
        if isinstance(gt, list):
            gt = np.array(gt) if len(gt) != 0 else np.zeros((0,10,10))

        assert (gt.ndim == dimension) or (gt.ndim == dimension+1) or gt.shape[0] == 0
        assert (pd.ndim == dimension) or (pd.ndim == dimension+1) or pd.shape[0] == 0

        self.ndim = dimension
        self.mode = mode
        self.tolerance = tolerance
        # self.is_label_map = (pd.ndim == dimension)

        if pd.ndim == dimension and gt.ndim == dimension:
            self.gt, self.pd = gt.astype(np.uint16), pd.astype(np.uint16)
            objs_gt = regionprops(self.gt)
            self.area_gt = {obj.label: obj.area for obj in objs_gt}
            objs_pd = regionprops(self.pd)
            self.area_pd = {obj.label: obj.area for obj in objs_pd}
        else:
            if pd.ndim == dimension:
                pd = map2stack(pd)
            if gt.ndim == dimension:
                gt = map2stack(gt)
            self.gt, self.pd = gt > 0, pd > 0
            self.area_gt = np.sum(self.gt, axis=tuple(range(1, 1+dimension)))
            self.gt = self.gt[self.area_gt > 0]
            self.area_gt = {l: c for l, c in enumerate(self.area_gt[self.area_gt>0])}
            self.area_pd = np.sum(self.pd, axis=tuple(range(1, 1+dimension)))
            self.pd = self.pd[self.area_pd > 0]
            self.area_pd = {l: c for l, c in enumerate(self.area_pd[self.area_pd>0])}

        self.num_gt, self.num_pd = len(self.area_gt), len(self.area_pd)

        # the max-overlap match is not symmetric, thus, store them separately
        self.match_pd = None  # (prediction label)-(matched gt label)
        self.intersection_pd = None # (prediction label)-(intersection area)
        self.match_gt = None # (gt label)-(matched prediction label)
        self.intersection_gt = None # (gt label)-(intersection area)

        # precision 
        self.precision_pd, self.precision_gt = None, None
        # recall
        self.recall_pd, self.recall_gt = None, None
        # F1 score
        self.f1_pd, self.f1_gt = None, None
        # dice
        self.dice_pd, self.dice_gt = None, None
        # jaccard
        self.jaccard_pd, self.jaccard_gt = None, None

        # aggreated area
        self.agg_intersection = None
        self.agg_union = None
        self.agg_area = None
        # match count, computed with respect to ground truth (which makes sense)
        self.gt_match_count = {}
        self.pd_match_count = {}
    

    def _match(self, subject='pred'):
        '''
        find the best match of each instance in prediction / ground truth
        Args:
            subject: 'pred' or 'gt'
        Return:
            match: dict of the best match
            intersection: dict of the intersection area
        '''
        if subject == 'pred' and self.match_pd is None:
            sub, ref = self.pd, self.gt
        elif subject == 'gt' and self.match_gt is None:
            sub, ref = self.gt, self.pd
        else:
            return None
        
        match, intersection = {}, {}

        if self.mode == "area":
            
            if sub.ndim == self.ndim:
                for r in regionprops(sub):
                    overlap = ref[tuple(r.coords[:,i] for i in range(self.ndim))]    
                    overlap = overlap[np.nonzero(overlap)]
                    if len(overlap) == 0:
                        match[r.label], intersection[r.label] = None, 0
                    else:
                        values, counts = np.unique(overlap, return_counts=True)
                        ind = np.argmax(counts)
                        match[r.label], intersection[r.label] = values[ind], counts[ind]
            else:
                for i in range(len(sub)):
                    overlap = np.sum(np.multiply(ref, np.expand_dims(sub[i], axis=0)), axis=tuple(range(1, 1+self.ndim)))
                    ind = np.argsort(overlap, kind='mergesort')
                    if len(ind) == 0 or overlap[ind[-1]] == 0:
                        match[i], intersection[i] = None, 0
                    else:
                        match[i], intersection[i] = ind[-1], overlap[ind[-1]]
        
        elif self.mode == "centroid":

            r_sub = regionprops(sub)
            label_sub = [r.label for r in r_sub] 
            pt_sub = np.array([r.centroid for r in r_sub])
            
            r_ref = regionprops(ref)
            label_ref = [r.label for r in r_ref] 
            pt_ref = np.array([r.centroid for r in r_ref])

            bpGraph = distance_matrix(pt_sub, pt_ref) < self.tolerance
            _, match_gp = GFG(bpGraph).maxBPM()

            match, intersection = {l: None for l in label_sub}, {l: 0 for l in label_sub}
            for i_ref, i_sub in enumerate(match_gp):
                if i_sub != -1:
                    match[label_sub[i_sub]] = label_ref[i_ref]
                    intersection[label_sub[i_sub]] = 1

        # elif self.mode == "area":

        #     if self.pd.ndim == self.ndim:
        #         for r_sub in regionprops(sub):    
        #             overlap, labels = [], []
        #             for r_ref in regionprops(sub):
        #                 bpGraph = distance_matrix(r_sub.coords, r_ref.coords) < self.tolerance
        #                 match_num, _ = GFG(bpGraph).maxBPM()
        #                 overlap.append(match_num)
        #                 labels.append(r_ref.label)
        #             ind = np.argsort(np.array(overlap), kind='mergesort')
        #             if overlap[ind[-1]] == 0:
        #                 match[r_sub.label], intersection[r_sub.label] = None, 0
        #             else:
        #                 match[r_sub.label], intersection[r_sub.label] = labels[ind[-1]], overlap[ind[-1]]
        #     else:    
        #         overlap = []
        #         for i_sub in range(len(sub)):
        #             pts_sub = np.transpose(np.array(np.nonzero(sub[i_sub])))
        #             for i_ref in range(len(ref)):
        #                 pts_ref = np.transpose(np.array(np.nonzero(ref[i_ref])))
        #                 bpGraph = distance_matrix(pts_sub, pts_ref) < self.tolerance
        #                 match_num, _ = GFG(bpGraph).maxBPM()
        #                 overlap.append(match_num)
        #             ind = np.argsort(np.array(overlap), kind='mergesort')
        #             if overlap[ind[-1]] == 0:
        #                 match[i_sub], intersection[i_sub] = None, 0
        #             else:
        #                 match[i_sub], intersection[i_sub] = ind[-1], overlap[ind[-1]]

        if subject == 'pred':
            self.match_pd, self.intersection_pd = match, intersection
        else:
            self.match_gt, self.intersection_gt = match, intersection
    

    def _getSegPrecision(self, subject='pred'):

        '''
        compute the segmentation precision of each 'subject' object
        '''
        if subject == 'pred' and self.precision_pd is None:    
            self._match('pred')
            self.precision_pd = {k: self.intersection_pd[k] / self.area_pd[k] for k in self.match_pd.keys()}

        if subject == 'gt' and self.precision_gt is None:    
            self._match('gt')
            self.precision_gt = {k: self.intersection_gt[k] / self.area_gt[k] for k in self.match_gt.keys()}


    def _getSegRecall(self, subject='pred'):

        '''
        compute the segmentation recall of each 'subject' object
        '''
        if subject == 'pred' and self.recall_pd is None:    
            self._match('pred')
            self.recall_pd = {}
            for k, m in self.match_pd.items():
                self.recall_pd[k] = self.intersection_pd[k] / self.area_gt[m] if m is not None else 0

        if subject == 'gt' and self.recall_gt is None:    
            self._match('gt')
            self.recall_gt = {}
            for k, m in self.match_gt.items():
                self.recall_gt[k] = self.intersection_gt[k] / self.area_pd[m] if m is not None else 0


    def _getSegF1(self, subject='pred'):
        '''
        compute the segmentation f1 score of each 'subject' object
        '''
        self._getSegPrecision(subject)
        self._getSegRecall(subject)

        if subject == 'pred' and self.f1_pd is None:
            self.f1_pd = {}
            for k, p in self.precision_pd.items():
                self.f1_pd[k] = 2*(p*self.recall_pd[k])/(p + self.recall_pd[k] + 1e-8)

        if subject == 'gt' and self.f1_gt is None:
            self.f1_gt = {}
            for k, p in self.precision_gt.items():
                self.f1_gt[k] = 2*(p*self.recall_gt[k])/(p + self.recall_gt[k] + 1e-8)
    

    def _getSegJaccard(self, subject='pred'):

        '''
        compute the segmentation Jaccard index of each 'subject' object
        '''
        self._match(subject)
        
        if subject == 'pred' and self.jaccard_pd is None:
            match, intersection = self.match_pd, self.intersection_pd
            area_sub, area_ref = self.area_pd, self.area_gt
        elif subject == 'gt' and self.jaccard_gt is None:
            match, intersection = self.match_gt, self.intersection_gt
            area_sub, area_ref = self.area_gt, self.area_pd
        else:
            return None

        jaccard = {}
        for k, m in match.items():
            union = area_sub[k] - intersection[k]
            if m is not None:
                union += area_ref[m]
            jaccard[k] = intersection[k] / union
        
        if subject == 'pred':
            self.jaccard_pd = jaccard
        else:
            self.jaccard_gt = jaccard


    def _getSegDice(self, subject='pred'):
        
        '''
        compute the segmentation Dice index of each 'subject' object
        '''
        self._match(subject)

        if subject == 'pred' and self.dice_pd is None:
            match, intersection = self.match_pd, self.intersection_pd
            area_sub, area_ref = self.area_pd, self.area_gt
        elif subject == 'gt' and self.dice_gt is None:
            match, intersection = self.match_gt, self.intersection_gt
            area_sub, area_ref = self.area_gt, self.area_pd
        else:
            return None

        dice = {}
        for k, m in match.items():
            agg_area = area_sub[k] + area_ref[m] if m is not None else area_sub[k]
            dice[k] = 2 * intersection[k] / agg_area
        
        if subject == 'pred':
            self.dice_pd = dice
        else:
            self.dice_gt = dice
        

    def averageSegPrecision(self, subject='pred'):

        '''
        average of the segmentation precision of each 'subject' object
        '''
        if self.mode == 'centroid':
            raise Exception("averageSegPrecision is not a valid score in 'centroid' mode")

        self._getSegPrecision(subject)
        if subject == 'pred':
            return np.mean(list(self.precision_pd.values()))
        else:
            return np.mean(list(self.precision_gt.values()))


    def averageSegRecall(self, subject='pred'):

        '''
        average of the segmentation recall of each 'subject' object
        '''
        if self.mode == 'centroid':
            raise Exception("averageSegRecall is not a valid score in 'centroid' mode")
        
        self._getSegRecall(subject)
        if subject == 'pred':
            return np.mean(list(self.recall_pd.values()))
        else:
            return np.mean(list(self.recall_gt.values())) 
             

    def averageSegF1(self, subject='pred'):
        '''
        average of the segmentation F1 score of each 'subject' object
        '''
        if self.mode == 'centroid':
            raise Exception("averageSegF1 is not a valid score in 'centroid' mode")
        
        self._getSegF1(subject)
        if subject == 'pred':
            return np.mean(list(self.f1_pd.values()))
        else:
            return np.mean(list(self.f1_gt.values()))
        

    def averageJaccard(self, subject='pred'):
        '''
        average of the segmentation Jaccard index of each 'subject' object
        '''
        if self.mode == 'centroid':
            raise Exception("averageJaccard is not a valid score in 'centroid' mode")
    
        self._getSegJaccard(subject)
        if subject == 'pred':
            return np.mean(list(self.jaccard_pd.values()))
        else:
            return np.mean(list(self.jaccard_gt.values()))


    def averageDice(self, subject='pred'):
        '''
        average of the segmentation dice of each 'subject' object
        '''
        if self.mode == 'centroid':
            raise Exception("averageDice is not a valid score in 'centroid' mode")

        self._getSegDice(subject)
        if subject == 'pred':
            return np.mean(list(self.dice_pd.values()))
        else:
            return np.mean(list(self.dice_gt.values()))


    def accumulate_area(self):
        '''  
        Reference:
            A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology
        '''
        if self.agg_intersection is None or self.agg_area is None or self.agg_union is None:
            self.agg_intersection, self.agg_union, self.agg_area = 0, 0, 0
            self._match('gt')
            matched_pd = []
            for k, m in self.match_gt.items():
                self.agg_intersection += self.intersection_gt[k]
                self.agg_union += (self.area_gt[k] - self.intersection_gt[k])
                self.agg_area += self.area_gt[k]
                if m is not None:
                    self.agg_union += self.area_pd[m]
                    self.agg_area += self.area_pd[m]
                    matched_pd.append(m)
            # add the area of not matched predictions
            agg_ex = np.sum(list(self.area_pd.values()))
            for l in np.unique(matched_pd):
                agg_ex -= self.area_pd[l]
            self.agg_union += agg_ex
            self.agg_area += agg_ex
        return self.agg_intersection, self.agg_union, self.agg_area


    def aggregatedJaccard(self):
        '''  
        Reference:
            A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology
        '''

        if self.mode == 'centroid':
            raise Exception("aggregatedJaccard is not a valid score in 'centroid' mode")

        agg_intersection, agg_union, _ = self.accumulate_area()
        if agg_intersection == 0 and agg_union == 0:
            return 1
        else:
            return agg_intersection/agg_union
    
    def AJI(self): # alias of aggregatedJaccard (aggregated Jaccard index)
        return self.aggregatedJaccard()


    def aggregatedDice(self):
        ''' 
        no defination found, derived from aggrated Jaccard Index
        Reference:
            CNN-BASED PREPROCESSING TO OPTIMIZE WATERSHED-BASED CELL SEGMENTATION IN 3D CONFOCAL MICROSCOPY IMAGES
        '''

        if self.mode == 'centroid':
            raise Exception("aggregatedDice is not a valid score in 'centroid' mode")

        agg_intersection, _, agg_area = self.accumulate_area()
        if agg_intersection == 0 and agg_union == 0:
            return 1
        else:
            return 2*agg_intersection/agg_area
    
    def ADS(self): # alias of aggregatedDice (aggregated Dice score)
        return self.aggregatedJaccard()


    def SBD(self):
        '''
        symmetric best dice
        '''
        if self.mode == 'centroid':
            raise Exception("SBD is not a valid score in 'centroid' mode")
        return min(self.averagedDice('pred'), self.averagedDice('gt'))


    def match_num(self, thres, metric='Jaccard', subject='gt'):
        '''
        Args:
            thres: threshold to determine the a match
            metric: metric used to determine match, 'Jaccard' or 'Dice'
        Retrun:
            match_count, gt_count: the number of matches, the number of matched gt objects
        '''
        match_count = self.gt_match_count if subject == 'gt' else self.pd_match_count
        self._match(subject)
        match = self.match_gt if subject == 'gt' else self.match_pd

        if thres not in match_count.keys():
            count, count_c = 0, []
            if self.mode == 'centroid':
                for sub, ref in match.items():
                    if ref is not None:
                        count += 1
                        count_c.append(ref)
            else:
                if metric.lower() == 'f1':
                    self._getSegF1(subject)
                    score = self.f1_gt if subject == 'gt' else self.f1_pd
                elif metric.lower() == 'jaccard':
                    self._getSegJaccard(subject)
                    score = self.jaccard_gt if subject == 'gt' else self.jaccard_pd
                elif metric.lower() == 'dice':
                    self._getSegDice(subject)
                    score = self.dice_gt if subject == 'gt' else self.dice_pd
                for k, s in score.items():
                    if s >= thres:
                        count += 1
                        count_c.append(match[k])
            match_count[thres] = [count, len(np.unique(count_c))] 
            if subject == 'gt':
                self.gt_match_count = match_count
            else: 
                self.pd_match_count = match_count

        return match_count[thres]


    def detectionRecall(self, thres=0.5, metric='Jaccard'):
        pd_match_gt_count = self.match_num(thres=thres, metric=metric, subject='pred')[1]
        # it is possible that gt, pred are both empty (empty image)
        S = pd_match_gt_count/self.num_gt if self.num_gt > 0 else 1
        return S


    def detectionPrecision(self, thres=0.5, metric='Jaccard'):
        pd_match_pd_count = self.match_num(thres=thres, metric=metric, subject='gt')[0]
        # it is possible that gt, pred are both empty
        A = pd_match_pd_count/self.num_pd if self.num_pd > 0 else 1
        return A


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
            return np.mean(dps)

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
            return np.mean(fnr)

    def P_DSB(self, thres=0.5, metric='Jaccard'):
        '''
        the precision based on Data Scient Bowl 2018 definition: https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''
        gt_match_count = self.match_num(thres=thres, metric=metric, subject='gt')
        union = self.num_gt + self.num_pd - gt_match_count[1]
        # it is possible that gt, pred are both empty
        P = gt_match_count[0]/union if union > 0 else 1
        return P


    def AP_DSB(self, thres=None, metric='Jaccard', interpolated=False):
        '''
        average precision based on Data Scient Bowl 2018 definition: https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''
        if self.mode == 'centroid':
            return self.P_DSB()
        else:
            thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] if thres is None else thres
            ps = [self.P_DSB(thres=t, metric=metric) for t in thres]
            if interpolated:
                ps = [max(ps[i:-1]) for i in range(len(ps)-1)] + [ps[-1]]
            return np.mean(ps)


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
        self.total_pd += e.num_pd
        self.total_gt += e.num_gt
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

        match_count, num_gt = 0, 0
        for e in self.examples:
            pd_match_gt_count = e.match_num(thres=thres, metric=metric, subject='pred')[1]
            match_count += pd_match_gt_count
            num_gt += e.num_gt
        S = match_count/num_gt if num_gt > 0 else 1
        if self.verbose:
            print("detectionRecall over the whole dataset under '" + metric + "' {}: {}".format(thres, S))
        return S

    def detectionPrecision(self, thres=0.5, metric='Jaccard'):

        match_count, num_pd = 0, 0
        for e in self.examples:
            pd_match_pd_count = e.match_num(thres=thres, metric=metric, subject='pred')[0]
            match_count += pd_match_pd_count
            num_pd += e.num_pd
        A = match_count/num_pd if num_pd > 0 else 1
        if self.verbose:
            print("detectionPrecision over the whole dataset under '" + metric + "' {}: {}".format(thres, A))
        return A


    def P_DSB(self, thres=0.5, metric='Jaccard'):
        '''
        the precision based on Data Scient Bowl 2018 definition: https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''

        match_count_gt, match_count_pd, num_pd, num_gt = 0, 0, 0, 0
        for e in self.examples:
            match_count = e.match_num(thres=thres, metric=metric, subject='gt')
            match_count_gt += match_count[0]
            match_count_pd += match_count[1]
            num_pd += e.num_pd
            num_gt += e.num_gt
            # print('===================')
            # print(match_count[0], match_count[1], e.num_pd, e.num_gt)
            # print(match_count_gt, match_count_pd, num_pd, num_gt)
        union = num_gt + num_pd - match_count_pd
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

    #### toy test ####
    # gt = imread('./test/toy_example/gt.png')
    # pred = imread('./test/toy_example/pred.png')
    # sample = Sample(pred, gt, mode='area')
    # subject = 'gt'
    # sample._match(subject=subject)
    # # print(sample.match_pd, sample.intersection_pd, sample.match_gt, sample.intersection_gt)
    # print('averageSegPrecision', sample.averageSegPrecision(subject))
    # print('averageSegRecall', sample.averageSegRecall(subject))
    # print('averageSegF1', sample.averageSegF1(subject))
    # print('averageJaccard', sample.averageJaccard(subject))
    # print('averageDice', sample.averageDice(subject))
    # print('aggregatedJaccard', sample.aggregatedJaccard())
    # print('aggregatedDice', sample.aggregatedDice())
    # print('COCO AP', sample.AP_COCO())
    # print('average false negative ratio', sample.AFNR())
    # print('DSB AP', sample.AP_DSB())



    f_gts = sorted(glob.glob('./test/cell/gt/*.png'))[0:50]
    f_preds = sorted(glob.glob('./test/cell/pred/*.tif'))[0:50]

    # evalation of a whole dataset
    e = Evaluator(dimension=2, mode='area')
    for f_gt, f_pred in zip(f_gts, f_preds):
        pred = imread(f_pred)
        gt = imread(f_gt)
        # add one segmentation
        e.add_example(pred, gt)

    e.detectionRecall()
    e.detectionPrecision()
    e.AP_COCO()
    e.AP_DSB()
    e.AFNR()
    e.AJI()
    e.ADS()