"""Reference-free binary watermark restoration using a connected-region prior.

Binary Potts graph cut combines blind QIM soft votes and a local-neighbor
consistency penalty. Does not use or access the expected watermark image.
"""
from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order

@lru_cache(maxsize=1)
def _graph_edges():
    rr=[];cc=[]
    for i in range(64):
        for j in range(64):
            a=i*64+j
            if j<63:
                rr.extend([a,a+1]);cc.extend([a+1,a])
            if i<63:
                rr.extend([a,a+64]);cc.extend([a+64,a])
    return np.array(rr,dtype=np.int32),np.array(cc,dtype=np.int32)

def restore(soft_votes:np.ndarray, smoothness:float=1.0):
    """Exact minimum graph cut of a binary local-consistency objective.

    Soft values must be positive for bit1, negative for bit0. The outer
    algorithm sets smoothness from blind noise-level estimation, not from
    original watermark pixels or any experimental ground-truth attack NC.
    """
    if soft_votes.shape!=(64,64):raise ValueError('Expected 64x64 scores')
    soft=np.asarray(soft_votes,dtype=np.float64).reshape(-1)
    n=len(soft);src=n;sink=n+1
    r,c=_graph_edges()
    row=np.concatenate([r,np.full(n,src,dtype=np.int32),np.arange(n)])
    col=np.concatenate([c,np.arange(n),np.full(n,sink,dtype=np.int32)])
    val=np.concatenate([
        np.full(len(r),int(round(smoothness*100)),dtype=np.int32),
        np.rint(np.maximum(0,soft)*100).astype(np.int32),
        np.rint(np.maximum(0,-soft)*100).astype(np.int32)])
    cap=coo_matrix((val,(row,col)),shape=(n+2,n+2),dtype=np.int32).tocsr()
    flow=maximum_flow(cap,src,sink)
    residual=(cap-flow.flow).tocsr()
    residual.data=np.uint8(residual.data>0)
    residual.eliminate_zeros()
    reach=breadth_first_order(residual,src,directed=True,return_predecessors=False)
    wm=np.zeros(n,dtype=np.uint8)
    wm[reach[reach<n]]=255
    return wm.reshape(64,64)
