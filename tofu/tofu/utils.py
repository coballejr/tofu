import numpy as np

def diag_to_array(data):
    dataset, num_diag = [], len(data["0"].keys())
    for dim in data.keys():
        X = []
        for diag in range(num_diag):
            pers_diag = np.array(data[dim][str(diag)])
            X.append(pers_diag)
        dataset.append(X)
    return dataset

def diag_to_dict(D):
    X = dict()
    for f in D.keys():
        df = diag_to_array(D[f])
        for dim in range(len(df)):
            X[str(dim) + "_" + f] = df[dim]
    return X

def pad(dgm,max_feats):
    if not dgm.shape[0]:
        return 100*np.ones((max_feats,2))
    
    n_feats = dgm.shape[0]
    return np.pad(dgm,((0,max_feats-n_feats),(0,0)),constant_values=((100,100),(0,0)))

def scale_persistence(dgm):
    pers = dgm[:,1]
    scaled_pers = pers/np.sum(pers)
    return np.hstack((dgm[:,0].reshape((-1,1)),scaled_pers.reshape((-1,1))))

def motion_capture_pd_dim(multi_dim_pd_lst, dim):
    return [dgm[dim] for dgm in multi_dim_pd_lst]

def motion_capture_pad(dgms):
    max_feats = np.max([dgm[0].shape[0] for dgm in dgms])
    return np.array([pad(dgm[0],max_feats) for dgm in dgms])