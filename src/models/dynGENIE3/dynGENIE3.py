from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from numpy import *
from numpy.random import permutation, uniform
import time
from operator import itemgetter
from multiprocessing import Pool
from scipy.stats import pearsonr
from itertools import combinations

from sklearn.utils import check_random_state, compute_sample_weight
from warnings import catch_warnings, simplefilter, warn
from numbers import Integral, Real
# from sklearn.utils.parallel import delayed, Parallel
from joblib import Parallel
from sklearn.utils.fixes import delayed

class OptimizedRandomForestRegressor(RandomForestRegressor):
    MAX_INT = iinfo(int32).max

    # Copied from sklearn
    def _get_n_samples_bootstrap(self, n_samples, max_samples):
        """
        Get the number of samples in a bootstrap sample.

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.
        max_samples : int or float
            The maximum number of samples to draw from the total available:
                - if float, this indicates a fraction of the total and should be
                the interval `(0.0, 1.0]`;
                - if int, this indicates the exact number of samples;
                - if None, this indicates the total number of samples.

        Returns
        -------
        n_samples_bootstrap : int
            The total number of samples to draw for the bootstrap sample.
        """
        if max_samples is None:
            return n_samples

        if isinstance(max_samples, Integral):
            if max_samples > n_samples:
                msg = "`max_samples` must be <= n_samples={} but got value {}"
                raise ValueError(msg.format(n_samples, max_samples))
            return max_samples

        if isinstance(max_samples, Real):
            return round(n_samples * max_samples)

    # Copied from sklearn
    def _generate_sample_indices(self, random_state, n_samples, n_samples_bootstrap):
        """
        Private function used to _parallel_build_trees function."""

        random_instance = check_random_state(random_state)
        sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)

        return sample_indices

    # Copied from sklearn
    def _generate_unsampled_indices(self, random_state, n_samples, n_samples_bootstrap):
        """
        Private function used to forest._set_oob_score function."""
        sample_indices = self._generate_sample_indices(
            random_state, n_samples, n_samples_bootstrap
        )
        sample_counts = bincount(sample_indices, minlength=n_samples)
        unsampled_mask = sample_counts == 0
        indices_range = arange(n_samples)
        unsampled_indices = indices_range[unsampled_mask]

        return unsampled_indices

    # Copied from sklearn
    def _parallel_build_trees(
        self, 
        tree,
        bootstrap,
        X,
        y,
        sample_weight,
        tree_idx,
        n_trees,
        verbose=0,
        class_weight=None,
        n_samples_bootstrap=None,
    ):
        """
        Private function used to fit a single tree in parallel."""
        if verbose > 1:
            print("building tree %d of %d" % (tree_idx + 1, n_trees))

        if bootstrap:
            n_samples = X.shape[0]
            if sample_weight is None:
                curr_sample_weight = ones((n_samples,), dtype=float64)
            else:
                curr_sample_weight = sample_weight.copy()

            indices = self._generate_sample_indices(
                tree.random_state, n_samples, n_samples_bootstrap
            )
            sample_counts = bincount(indices, minlength=n_samples)
            curr_sample_weight *= sample_counts

            if class_weight == "subsample":
                with catch_warnings():
                    simplefilter("ignore", DeprecationWarning)
                    curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
            elif class_weight == "balanced_subsample":
                curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

            tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
        else:
            tree.fit(X, y, sample_weight=sample_weight, check_input=False)

        return tree

    def _parallel_build_trees_and_return_importance(self, *args, **kwargs):
        tree = self._parallel_build_trees(*args, **kwargs)
        return(tree.tree_.compute_feature_importances(normalize=False))

    # Copied based on sklearn
    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()

        y = atleast_1d(y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        # if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
        #     y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = self._get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._validate_estimator()
        #if isinstance(self, (RandomForestRegressor, ExtraTreesRegressor)):
            # TODO(1.3): Remove "auto"
        #    if self.max_features == "auto":
        #        warn(
        #            "`max_features='auto'` has been deprecated in 1.1 "
        #            "and will be removed in 1.3. To keep the past behaviour, "
        #            "explicitly set `max_features=1.0` or remove this "
        #            "parameter as it is also the default value for "
        #            "RandomForestRegressors and ExtraTreesRegressors.",
        #            FutureWarning,
        #        )

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(self.MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(self._parallel_build_trees_and_return_importance)( # changed here
                    t,
                    self.bootstrap,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self


def compute_feature_importances(estimator):
    
    """Computes variable importances from a trained tree-based model.
    """
    
    if isinstance(estimator, BaseDecisionTree):
        return estimator.tree_.compute_feature_importances(normalize=False)
    else:
        # importances = [e.tree_.compute_feature_importances(normalize=False)
        #                for e in estimator.estimators_]
        importances = estimator.estimators_
        importances = array(importances)
        return sum(importances,axis=0) / len(estimator)



def get_link_list(VIM,gene_names=None,regulators='all',maxcount='all',file_name=None):
    
    """Gets the ranked list of (directed) regulatory links.
    
    Parameters
    ----------
    
    VIM: numpy array
        Array as returned by the function dynGENIE3(), in which the element (i,j) is the score of the edge directed from the i-th gene to the j-th gene. 
        
    gene_names: list of strings, optional
        List of length p, where p is the number of rows/columns in VIM, containing the names of the genes. The i-th item of gene_names must correspond to the i-th row/column of VIM. When the gene names are not provided, the i-th gene is named Gi.
        default: None
        
    regulators: list of strings, optional
        List containing the names of the candidate regulators. When a list of regulators is provided, the names of all the genes must be provided (in gene_names), and the returned list contains only edges directed from the candidate regulators. When regulators is set to 'all', any gene can be a candidate regulator.
        default: 'all'
        
    maxcount: 'all' or positive integer, optional
        Writes only the first maxcount regulatory links of the ranked list. When maxcount is set to 'all', all the regulatory links are written.
        default: 'all'
        
    file_name: string, optional
        Writes the ranked list of regulatory links in the file file_name.
        default: None
        
        
    
    Returns
    -------
    
    The list of regulatory links, ordered according to the edge score. Auto-regulations do not appear in the list. Regulatory links with a score equal to zero are randomly permuted. In the ranked list of edges, each line has format:
        
        regulator   target gene     score of edge
    """
    
    # Check input arguments      
    if not isinstance(VIM,ndarray):
        raise ValueError('VIM must be a square array')
    elif VIM.shape[0] != VIM.shape[1]:
        raise ValueError('VIM must be a square array')
        
    ngenes = VIM.shape[0]
        
    if gene_names is not None:
        if not isinstance(gene_names,(list,tuple)):
            raise ValueError('input argument gene_names must be a list of gene names')
        elif len(gene_names) != ngenes:
            raise ValueError('input argument gene_names must be a list of length p, where p is the number of columns/genes in the expression data')
        
    if regulators != 'all':
        if not isinstance(regulators,(list,tuple)):
            raise ValueError('input argument regulators must be a list of gene names')

        if gene_names is None:
            raise ValueError('the gene names must be specified (in input argument gene_names)')
        else:
            sIntersection = set(gene_names).intersection(set(regulators))
            if not sIntersection:
                raise ValueError('The genes must contain at least one candidate regulator')
        
    if maxcount != 'all' and not isinstance(maxcount,int):
        raise ValueError('input argument maxcount must be "all" or a positive integer')
        
    if file_name is not None and not isinstance(file_name,str):
        raise ValueError('input argument file_name must be a string')
    
    

    # Get the indices of the candidate regulators
    if regulators == 'all':
        input_idx = list(range(ngenes))
    else:
        input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
               
    nTFs = len(input_idx)
    
    # Get the non-ranked list of regulatory links
    vInter = [(i,j,score) for (i,j),score in ndenumerate(VIM) if i in input_idx and i!=j]
    
    # Rank the list according to the weights of the edges        
    vInter_sort = sorted(vInter,key=itemgetter(2),reverse=True)
    nInter = len(vInter_sort)
    
    # Random permutation of edges with score equal to 0
    flag = 1
    i = 0
    while flag and i < nInter:
        (TF_idx,target_idx,score) = vInter_sort[i]
        if score == 0:
            flag = 0
        else:
            i += 1
            
    if not flag:
        items_perm = vInter_sort[i:]
        items_perm = random.permutation(items_perm)
        vInter_sort[i:] = items_perm
        
    # Write the ranked list of edges
    nToWrite = nInter
    if isinstance(maxcount,int) and maxcount >= 0 and maxcount < nInter:
        nToWrite = maxcount
        
    if file_name:
    
        outfile = open(file_name,'w')
    
        if gene_names is not None:
            for i in range(nToWrite):
                (TF_idx,target_idx,score) = vInter_sort[i]
                TF_idx = int(TF_idx)
                target_idx = int(target_idx)
                outfile.write('%s\t%s\t%.6f\n' % (gene_names[TF_idx],gene_names[target_idx],score))
        else:
            for i in range(nToWrite):
                (TF_idx,target_idx,score) = vInter_sort[i]
                TF_idx = int(TF_idx)
                target_idx = int(target_idx)
                outfile.write('G%d\tG%d\t%.6f\n' % (TF_idx+1,target_idx+1,score))
            
        
        outfile.close()
        
    else:
        
        if gene_names is not None:
            for i in range(nToWrite):
                (TF_idx,target_idx,score) = vInter_sort[i]
                TF_idx = int(TF_idx)
                target_idx = int(target_idx)
                print('%s\t%s\t%.6f' % (gene_names[TF_idx],gene_names[target_idx],score))
        else:
            for i in range(nToWrite):
                (TF_idx,target_idx,score) = vInter_sort[i]
                TF_idx = int(TF_idx)
                target_idx = int(target_idx)
                print('G%d\tG%d\t%.6f' % (TF_idx+1,target_idx+1,score))
                
                
def estimate_degradation_rates(TS_data,time_points):
    
    """
    For each gene, the degradation rate is estimated by assuming that the gene expression x(t) follows:
    x(t) =  A exp(-alpha * t) + C_min,
    between the highest and lowest expression values.
    C_min is set to the minimum expression value over all genes and all samples.
    """
    
    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)
    
    C_min = TS_data[0].min()
    if nexp > 1:
        for current_timeseries in TS_data[1:]:
            C_min = min(C_min,current_timeseries.min())
    
    alphas = zeros((nexp,ngenes))
    
    for (i,current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]
        
        for j in range(ngenes):
            
            idx_min = argmin(current_timeseries[:,j])
            idx_max = argmax(current_timeseries[:,j])
            
            xmin = current_timeseries[idx_min,j]
            xmax = current_timeseries[idx_max,j]
            
            tmin = current_time_points[idx_min]
            tmax = current_time_points[idx_max]
            
            xmin = max(xmin-C_min,1e-6)
            xmax = max(xmax-C_min,1e-6)
                
            xmin = log(xmin)
            xmax = log(xmax)
            
            alphas[i,j] = (xmax - xmin) / abs(tmin - tmax)
                
    alphas = alphas.max(axis=0)
 
    return alphas
        

    
         


def dynGENIE3(TS_data,time_points,alpha='from_data',SS_data=None,gene_names=None,regulators='all',tree_method='RF',K='sqrt',ntrees=1000,compute_quality_scores=False,save_models=False,nthreads=1, **kwargs):

    '''Computation of tree-based scores for all putative regulatory links.

    Parameters
    ----------

    TS_data: list of numpy arrays
        List of arrays, where each array contains the gene expression values of one time series experiment. Each row of an array corresponds to a time point and each column corresponds to a gene. The i-th column of each array must correspond to the same gene.
    
    time_points: list of one-dimensional numpy arrays
        List of n vectors, where n is the number of time series (i.e. the number of arrays in TS_data), containing the time points of the different time series. The i-th vector specifies the time points of the i-th time series of TS_data.
    
    alpha: either 'from_data', a positive number or a vector of positive numbers
        Specifies the degradation rate of the different gene expressions. 
        When alpha = 'from_data', the degradation rate of each gene is estimated from the data, by assuming an exponential decay between the highest and lowest observed expression values.
        When alpha is a vector of positive numbers, the i-th element of the vector must specify the degradation rate of the i-th gene.
        When alpha is a positive number, all the genes are assumed to have the same degradation rate alpha.
        default: 'from_data'
    
    SS_data: numpy array, optional
        Array containing steady-state gene expression values. Each row corresponds to a steady-state condition and each column corresponds to a gene. The i-th column/gene must correspond to the i-th column/gene of each array of TS_data.
        default: None

    gene_names: list of strings, optional
        List of length p containing the names of the genes, where p is the number of columns/genes in each array of TS_data. The i-th item of gene_names must correspond to the i-th column of each array of TS_data (and the i-th column of SS_data when SS_data is not None).
        default: None
    
    regulators: list of strings, optional
        List containing the names of the candidate regulators. When a list of regulators is provided, the names of all the genes must be provided (in gene_names). When regulators is set to 'all', any gene can be a candidate regulator.
        default: 'all'
    
    tree-method: 'RF' or 'ET', optional
        Specifies which tree-based procedure is used: either Random Forest ('RF') or Extra-Trees ('ET')
        default: 'RF'
    
    K: 'sqrt', 'all' or a positive integer, optional
        Specifies the number of selected attributes at each node of one tree: either the square root of the number of candidate regulators ('sqrt'), the number of candidate regulators ('all'), or any positive integer.
        default: 'sqrt'
     
    ntrees: positive integer, optional
        Specifies the number of trees grown in an ensemble.
        default: 1000

    compute_quality_scores: boolean, optional
        Indicates if the scores assessing the edge ranking quality must be computed or not. These scores are:
        - the score of prediction of out-of-bag samples, i.e. the Pearson correlation between the predicted and true output values. To be able to compute this score, Random Forests must be used (i.e. parameter tree_method must be set to 'RF').
        - the stability score, measuring the average stability among the top-5 candidate regulators returned by each tree of a forest.
        default: False

    save_models: boolean, optional
        Indicates if the tree models (one for each gene) must be saved or not.

    nthreads: positive integer, optional
        Number of threads used for parallel computing
        default: 1
    
    
    Returns
    -------

    A tuple (VIM, alphas, prediction_score, stability_score, treeEstimators).

    VIM: array in which the element (i,j) is the score of the edge directed from the i-th gene to the j-th gene. All diagonal elements are set to zero (auto-regulations are not considered). When a list of candidate regulators is provided, all the edges directed from a gene that is not a candidate regulator are set to zero.
 
    alphas: vector in which the i-th element is the degradation rate of the i-th gene.
    
    prediction_score: prediction score on out-of-bag samples (averaged over all genes and all trees). prediction_score is an empty list if compute_quality_scores is set to False or if tree_method is not set to 'RF'.
    
    stability_score: stability score (averaged over all genes). stability_score is an empty list if compute_quality_scores is set to False.
    
    treeEstimators: list of tree models, where the i-th model is the model predicting the expression of the i-th gene. treeEstimators is an empty list if save_models is set to False.

    '''

    time_start = time.time()

    # Check input arguments
    if not isinstance(TS_data,(list,tuple)):
        raise ValueError('TS_data must be a list of arrays, where each row of an array corresponds to a time point/sample and each column corresponds to a gene')
    
    for expr_data in TS_data:
        if not isinstance(expr_data,ndarray):
            raise ValueError('TS_data must be a list of arrays, where each row of an array corresponds to a time point/sample and each column corresponds to a gene')
    
    ngenes = TS_data[0].shape[1]

    if len(TS_data) > 1:
        for expr_data in TS_data[1:]:
            if expr_data.shape[1] != ngenes:
                raise ValueError('The number of columns/genes must be the same in every array of TS_data.')
                
                
    if not isinstance(time_points,(list,tuple)):
        raise ValueError('time_points must be a list of n one-dimensional arrays, where n is the number of time series experiments in TS_data')
    
    if len(time_points) != len(TS_data):
        raise ValueError('time_points must be a list of n one-dimensional arrays, where n is the number of time series experiments in TS_data')
    
    for tp in time_points:
        if (not isinstance(tp,(list,tuple,ndarray))) or (isinstance(tp,ndarray) and tp.ndim > 1):
            raise ValueError('time_points must be a list of n one-dimensional arrays, where n is the number of time series in TS_data')
        
    for (i,expr_data) in enumerate(TS_data):
        if len(time_points[i]) != expr_data.shape[0]:
            raise ValueError('The length of the i-th vector of time_points must be equal to the number of rows in the i-th array of TS_data')

    if alpha != 'from_data':
        if not isinstance(alpha,(list,tuple,ndarray,int,float)):
            raise ValueError("input argument alpha must be either 'from_data', a positive number or a vector of positive numbers")
        
        if isinstance(alpha,(int,float)) and alpha < 0:
            raise ValueError("the degradation rate(s) specified in input argument alpha must be positive")
        
        if isinstance(alpha,(list,tuple,ndarray)):
            if isinstance(alpha,ndarray) and alpha.ndim > 1:
                raise ValueError("input argument alpha must be either 'from_data', a positive number or a vector of positive numbers")
            if len(alpha) != ngenes:
                raise ValueError('when input argument alpha is a vector, this must be a vector of length p, where p is the number of genes')
            for a in alpha:
                if a < 0:
                    raise ValueError("the degradation rate(s) specified in input argument alpha must be positive")

    if SS_data is not None:
        if not isinstance(SS_data,ndarray):
            raise ValueError('SS_data must be an array in which each row corresponds to a steady-state condition/sample and each column corresponds to a gene')
        
        if SS_data.ndim != 2:
            raise ValueError('SS_data must be an array in which each row corresponds to a steady-state condition/sample and each column corresponds to a gene')
    
        if SS_data.shape[1] != ngenes:
            raise ValueError('The number of columns/genes in SS_data must by the same as the number of columns/genes in every array of TS_data.')
        

    if gene_names is not None:
        if not isinstance(gene_names,(list,tuple)):
            raise ValueError('input argument gene_names must be a list of gene names')
        elif len(gene_names) != ngenes:
            raise ValueError('input argument gene_names must be a list of length p, where p is the number of columns/genes in the expression data')
    
    if regulators != 'all':
        if not isinstance(regulators,(list,tuple)):
            raise ValueError('input argument regulators must be a list of gene names')

        if gene_names is None:
            raise ValueError('the gene names must be specified (in input argument gene_names)')
        else:
            sIntersection = set(gene_names).intersection(set(regulators))
            if not sIntersection:
                raise ValueError('The genes must contain at least one candidate regulator')        
    
    if tree_method != 'RF' and tree_method != 'ET':
        raise ValueError('input argument tree_method must be "RF" (Random Forests) or "ET" (Extra-Trees)')
    
    if K != 'sqrt' and K != 'all' and not isinstance(K,int): 
        raise ValueError('input argument K must be "sqrt", "all" or a stricly positive integer')
    
    if isinstance(K,int) and K <= 0:
        raise ValueError('input argument K must be "sqrt", "all" or a stricly positive integer')

    if not isinstance(ntrees,int):
        raise ValueError('input argument ntrees must be a stricly positive integer')
    elif ntrees <= 0:
        raise ValueError('input argument ntrees must be a stricly positive integer')
    
    if not isinstance(compute_quality_scores,bool):
        raise ValueError('input argument compute_quality_scores must be a boolean (True or False)')
    
    if not isinstance(save_models,bool):
        raise ValueError('input argument save_models must be a boolean (True or False)')
        
    if not isinstance(nthreads,int):
        raise ValueError('input argument nthreads must be a stricly positive integer')
    elif nthreads <= 0:
        raise ValueError('input argument nthreads must be a stricly positive integer')
    
    

    
    # Re-order time points in increasing order
    for (i,tp) in enumerate(time_points):
        tp = array(tp, float32)
        indices = argsort(tp)
        time_points[i] = tp[indices]
        expr_data = TS_data[i]
        TS_data[i] = expr_data[indices,:]
        
    if alpha == 'from_data':
        alphas = estimate_degradation_rates(TS_data,time_points)
    elif isinstance(alpha,(int,float)):
        alphas = zeros(ngenes) + float(alpha)    
    else:
        alphas = [float(a) for a in alpha]

                
    print('Tree method: ' + str(tree_method))
    print('K: ' + str(K))
    print('Number of trees: ' + str(ntrees))
    print('alpha min: ' + str(min(alphas)))
    print('alpha max: ' + str(max(alphas)))
    print('\n')
                

    # Get the indices of the candidate regulators
    if regulators == 'all':
        input_idx = list(range(ngenes))
    else:
        input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]


    # Learn an ensemble of trees for each target gene and compute scores for candidate regulators
    VIM = zeros((ngenes,ngenes))

    if compute_quality_scores:
        if tree_method == 'RF':
            prediction_score = zeros(ngenes)
        else:
            prediction_score = []
        stability_score = zeros(ngenes)
    else:
        prediction_score = []
        stability_score = []
    
    if save_models:
        treeEstimators = [0] * ngenes
    else:
        treeEstimators = []

    for i in range(ngenes):
        print('Gene %d/%d...' % (i+1,ngenes))
    
        (vi,prediction_score_i,stability_score_i,treeEstimator) = dynGENIE3_single(TS_data,time_points,SS_data,i,alphas[i],input_idx,tree_method,K,ntrees,compute_quality_scores,save_models,nthreads,**kwargs)
        VIM[i,:] = vi
    
        if compute_quality_scores:
            if tree_method == 'RF':
                prediction_score[i] = prediction_score_i
            stability_score[i] = stability_score_i
        
        if save_models:
            treeEstimators[i] = treeEstimator

    VIM = transpose(VIM)
    if compute_quality_scores:
        if tree_method == 'RF':
            prediction_score = mean(prediction_score)
        stability_score = mean(stability_score)

    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM, alphas, prediction_score, stability_score, treeEstimators
    


def dynGENIE3_single(TS_data,time_points,SS_data,output_idx,alpha,input_idx,tree_method,K,ntrees,compute_quality_scores,save_models,nthreads, **kwargs):

    h = 1 # lag (in number of time points) used for the finite approximation of the derivative of the target gene expression
    ntop = 5 # number of top-ranked candidate regulators over which to compute the stability score

    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)
    nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data]) 
    ninputs = len(input_idx)

    # Construct learning sample 

    # Time-series data
    input_matrix_time = zeros((nsamples_time-h*nexp,ninputs))
    output_vect_time = zeros(nsamples_time-h*nexp)
    
    # Data for the computation of the prediction score on out-of-bag samples
    output_vect_time_present = zeros(nsamples_time-h*nexp)
    output_vect_time_future = zeros(nsamples_time-h*nexp)
    time_diff = zeros(nsamples_time-h*nexp)

    nsamples_count = 0

    for (i,current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]
        npoints = current_timeseries.shape[0]
        time_diff_current = current_time_points[h:] - current_time_points[:npoints-h]
        current_timeseries_input = current_timeseries[:npoints-h,input_idx]
        current_timeseries_output = (current_timeseries[h:,output_idx] - current_timeseries[:npoints-h,output_idx]) / time_diff_current + alpha*current_timeseries[:npoints-h,output_idx]
        nsamples_current = current_timeseries_input.shape[0]
        input_matrix_time[nsamples_count:nsamples_count+nsamples_current,:] = current_timeseries_input
        output_vect_time[nsamples_count:nsamples_count+nsamples_current] = current_timeseries_output
        output_vect_time_present[nsamples_count:nsamples_count+nsamples_current] = current_timeseries[:npoints-h,output_idx]
        output_vect_time_future[nsamples_count:nsamples_count+nsamples_current] = current_timeseries[h:,output_idx]
        time_diff[nsamples_count:nsamples_count+nsamples_current] = time_diff_current
        nsamples_count += nsamples_current
    
    # Steady-state data (if any)
    if SS_data is not None:

        input_matrix_steady = SS_data[:,input_idx]
        output_vect_steady = SS_data[:,output_idx] * alpha
    
        # Concatenation
        input_all = vstack([input_matrix_steady,input_matrix_time])
        output_all = concatenate((output_vect_steady,output_vect_time))
        
        del input_matrix_time
        del output_vect_time
        del input_matrix_steady
        del output_vect_steady
    
    else:
  
        input_all = input_matrix_time
        output_all = output_vect_time
        
        del input_matrix_time
        del output_vect_time


    # Parameters of the tree-based method

    # Whether or not to compute the prediction score of out-of-bag samples
    if compute_quality_scores and tree_method =='RF':
        oob_score = True
    else:
        oob_score = False
    
    # Parameter K of the tree-based method
    if (K == 'all') or (isinstance(K,int) and K >= len(input_idx)):
        max_features = "auto"
    else:
        max_features = K

    if tree_method == 'RF':
        treeEstimator = OptimizedRandomForestRegressor(n_estimators=ntrees,max_features=max_features,oob_score=oob_score,n_jobs=nthreads, **kwargs)
    elif tree_method == 'ET':
        treeEstimator = ExtraTreesRegressor(n_estimators=ntrees,max_features=max_features,oob_score=oob_score,n_jobs=nthreads)

    # Learn ensemble of trees
    treeEstimator.fit(input_all,output_all)

    # Compute importance scores
    feature_importances = compute_feature_importances(treeEstimator)
    vi = zeros(ngenes)
    vi[input_idx] = feature_importances
    # vi[output_idx] = 0

    # Normalize importance scores
    vi_sum = sum(vi)
    if vi_sum > 0:
        vi = vi / vi_sum
        
        
    # Ranking quality scores
    prediction_score_oob = []
    stability_score = []
        
    if compute_quality_scores:
        
        if tree_method == 'RF':
            
            # Prediction of out-of-bag samples
        
            if SS_data is not None:
            
                nsamples_SS = SS_data.shape[0]
            
                # Samples coming from the steady-state data
                oob_prediction_SS = treeEstimator.oob_prediction_[:nsamples_SS]
                output_pred_SS = oob_prediction_SS / alpha
            
                # Samples coming from the time series data
                oob_prediction_TS = treeEstimator.oob_prediction_[nsamples_SS:]
                output_pred_TS = (oob_prediction_TS - alpha*output_vect_time_present) * time_diff + output_vect_time_present
            
                output_pred = concatenate((output_pred_SS,output_pred_TS))
                output_true = concatenate((SS_data[:,output_idx],output_vect_time_future))
            
                (prediction_score_oob,tmp) = pearsonr(output_pred,output_true)
            
            else:
                oob_prediction_TS = treeEstimator.oob_prediction_
                output_pred_TS = (oob_prediction_TS - alpha*output_vect_time_present) * time_diff + output_vect_time_present
   
                (prediction_score_oob,tmp) = pearsonr(output_pred_TS,output_vect_time_future)
            
            
        # Stability score
   
        # Importances returned by each tree
        importances_by_tree = asarray(treeEstimator.estimators_)
        if output_idx in input_idx:
            idx = input_idx.index(output_idx)
            # Remove importances of target gene
            importances_by_tree = delete(importances_by_tree,idx,1)
            
        # Add some jitter to avoir numerical errors
        importances_by_tree = importances_by_tree + uniform(low=1e-12,high=1e-11,size=importances_by_tree.shape)
            
        if sum(importances_by_tree) > 0:
        
            # Ranking of candidate regulators
            ranking_by_tree = [importances_by_tree[i,:].argsort()[::-1] for i in range(ntrees)]
            top_by_tree = [set(r[:ntop]) for r in ranking_by_tree]
    
            # Stability score computed over the top-ranked candidate regulators
            stability_score = mean([len(top_by_tree[i].intersection(top_by_tree[j])) for (i,j) in combinations(range(ntrees),2)]) / float(ntop)
            
                
        # Variance of output is too small --> no forest was built and all the importances are zero    
        else:
            stability_score = 0.0
            
    if save_models: 
        return vi, prediction_score_oob, stability_score, treeEstimator
    else:
        return vi, prediction_score_oob, stability_score, []
    
    
    
    
    
    
    
    
def dynGENIE3_predict_doubleKO(expr_WT,treeEstimators,alpha,gene_names,regulators,KO1_gene,KO2_gene,nTimePoints,deltaT):
    
    '''Prediction of gene expressions in a double knockout experiment.

    Parameters
    ----------

    expr_WT: vector containing the gene expressions in the wild-type.
    
    treeEstimators: list of tree models, as returned by the function dynGENIE3(), where the i-th model is the model predicting the expression of the i-th gene. 
        The i-th model must correspond to the i-th gene in expr_WT.
    
    alpha: a positive number or a vector of positive numbers
        Specifies the degradation rate of the different gene expressions. 
        When alpha is a vector of positives, the i-th element of the vector must specify the degradation rate of the i-th gene.
        When alpha is a positive number, all the genes are assumed to have the same degradation rate.
    
    gene_names: list of strings
        List containing the names of the genes. The i-th item of gene_names must correspond to the i-th gene in expr_WT.
    
    regulators: list of strings
        List containing the names of the candidate regulators. When regulators is set to 'all', any gene can be a candidate regulator.
        Note that the candidate regulators must be the same as the ones used when calling the function dynGENIE3().
    
    KO1_gene: name of the first knocked-out gene.
    
    KO2_gene: name of the second knocked-out gene.
    
    nTimePoints: integer
        Specifies the number of time points for which to make a prediction.
    
    deltaT: a positive number
        Specifies the (constant) time interval between two predictions.
    
    
    
    Returns
    -------

    An array in which the element (t,i) is the predicted expression of the i-th gene at the t-th time point.
    The first row of the array contains the initial gene expressions (i.e. the expressions in expr_WT), where the expressions of the two knocked-out genes are set to 0.

    '''
    
    
    time_start = time.time()
    
    # Check input arguments
    if not isinstance(expr_WT,ndarray) or expr_WT.ndim > 1:
        raise ValueError("input argument expr_WT must be a vector of numbers")
        
    ngenes = len(expr_WT)
    
    if len(treeEstimators) != ngenes:
        raise ValueError("input argument treeEstimators must contain p tree models, where p is the number of genes in expr_WT")
    
    if not isinstance(alpha,(list,tuple,ndarray,int,float)):
        raise ValueError("input argument alpha must be a positive number or a vector of positive numbers")
        
    if isinstance(alpha,(int,float)) and alpha < 0:
        raise ValueError("the degradation rate(s) specified in input argument alpha must be positive")
        
    if isinstance(alpha,(list,tuple,ndarray)):
        if isinstance(alpha,ndarray) and alpha.ndim > 1:
            raise ValueError("input argument alpha must be a positive number or a vector of positive numbers")
        if len(alpha) != ngenes:
            raise ValueError('when input argument alpha is a vector, this must be a vector of length p, where p is the number of genes')
        for a in alpha:
            if a < 0:
                raise ValueError("the degradation rate(s) specified in input argument alpha must be positive")
     
    if not isinstance(gene_names,(list,tuple)):
        raise ValueError('input argument gene_names must be a list of gene names')
    elif len(gene_names) != ngenes:
        raise ValueError('input argument gene_names must be a list of length p, where p is the number of genes in expr_WT')

    if regulators != 'all':
        if not isinstance(regulators,(list,tuple)):
            raise ValueError('input argument regulators must be a list of gene names')

        sIntersection = set(gene_names).intersection(set(regulators))
        if not sIntersection:
            raise ValueError('The genes must contain at least one candidate regulator')
            
    if not (KO1_gene in gene_names):
        raise ValueError('input argument KO1_gene was not found in gene_names')
        
    if not (KO2_gene in gene_names):
        raise ValueError('input argument KO2_gene was not found in gene_names')

    if not isinstance(nTimePoints,int) or nTimePoints < 1:
        raise ValueError("input argument nTimePoints must be a strictly positive integer")
                   
    if not isinstance(deltaT,(int,float)) or deltaT < 0:
        raise ValueError("input argument deltaT must be a positive number")
        
    KO1_idx = gene_names.index(KO1_gene)
    KO2_idx = gene_names.index(KO2_gene)

    geneidx = list(range(ngenes))
    geneidx.remove(KO1_idx)
    geneidx.remove(KO2_idx)
    
    if isinstance(alpha,(int,float)):
        alphas = zeros(ngenes) + float(alpha)    
    else:
        alphas = [float(a) for a in alpha]
    
    # Get the indices of the candidate regulators
    if regulators == 'all':
        input_idx = list(range(ngenes))
    else:
        input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
                
        
    # Predict time series
    
    print('Predicting time series...')
       
    TS_predict = zeros((nTimePoints+1,ngenes))
    TS_predict[0,:] = expr_WT
    TS_predict[0,KO1_idx] = 0
    TS_predict[0,KO2_idx] = 0
    
    for t in range(1,nTimePoints+1):
        new_expr = [(treeEstimators[i].predict(TS_predict[t-1,input_idx].reshape(1,-1)) - alphas[i]*TS_predict[t-1,i]) * deltaT + TS_predict[t-1,i] for i in geneidx] 
        TS_predict[t,geneidx] = array(new_expr,float32).flatten()

    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return TS_predict

