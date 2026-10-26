import numpy as np
import cv2
from scipy.ndimage import binary_erosion

def grey(im,valid):
    x=np.nan_to_num(im.mean(0)).astype('float32');lo,hi=np.percentile(x[valid],[2,98])
    x=np.clip((x-lo)/max(hi-lo,1e-4),0,1)
    return cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply((x*255).astype('uint8'))

def estimate(a,b,ground,tr,minimum_spread=200):
    mask=binary_erosion(ground,iterations=3)
    sift=cv2.SIFT_create(nfeatures=8000,contrastThreshold=.01,edgeThreshold=12)
    ka,da=sift.detectAndCompute(a,mask.astype('uint8')*255);kb,db=sift.detectAndCompute(b,mask.astype('uint8')*255)
    rec=dict(source_features=len(ka),destination_features=len(kb));points=[]
    if da is None or db is None or min(len(da),len(db))<2:return dict(rec,status='insufficient_features'),points
    bf=cv2.BFMatcher(cv2.NORM_L2);ab=bf.knnMatch(da,db,k=2);ba=bf.knnMatch(db,da,k=2)
    reverse={(m.queryIdx,m.trainIdx) for v in ba if len(v)==2 for m,n in [v] if m.distance<.75*n.distance}
    matches=[m for v in ab if len(v)==2 for m,n in [v] if m.distance<.75*n.distance and (m.trainIdx,m.queryIdx) in reverse]
    rec['mutual_matches']=len(matches)
    if len(matches)<18:return dict(rec,status='insufficient_mutual_matches'),points
    pa=np.array([ka[m.queryIdx].pt for m in matches]);pb=np.array([kb[m.trainIdx].pt for m in matches]);diff=pb-pa
    east=tr.c+pa[:,0];north=tr.f-pa[:,1]
    fit=((np.floor(east/100)+np.floor(north/100)).astype('int64')%2)==0
    rec.update(fit_matches=int(fit.sum()),holdout_matches=int((~fit).sum()))
    if fit.sum()<12 or (~fit).sum()<6:return dict(rec,status='insufficient_spatial_split_matches'),points
    dd=diff[fit];votes=np.linalg.norm(dd[:,None,:]-dd[None,:,:],axis=2)<=1.5
    inside=votes[np.argmax(votes.sum(1))];delta=np.median(dd[inside],axis=0)
    for _ in range(2):inside=np.linalg.norm(dd-delta,axis=1)<=1.5;delta=np.median(dd[inside],axis=0)
    dx,dy=map(float,delta);res=np.linalg.norm(diff-delta,axis=1)
    train=fit&(res<=1.5);check=(~fit)&(res<=2)
    train_rms=float(np.sqrt(np.mean(res[train]**2)));check_rms=float(np.sqrt(np.mean(res[check]**2))) if check.any() else None
    spread=float(np.linalg.norm(np.ptp(pa[train],axis=0))) if train.any() else 0
    held_spread=float(np.linalg.norm(np.ptp(pa[check],axis=0))) if check.any() else 0
    rec.update(dx_pixel=dx,dy_pixel=dy,shift_east_m=dx,shift_north_m=-dy,
      fit_inliers=int(train.sum()),fit_inlier_fraction=float(train.sum()/fit.sum()),fit_RMSE_m=train_rms,fit_spread_m=spread,
      holdout_inliers=int(check.sum()),holdout_inlier_fraction=float(check.sum()/(~fit).sum()),holdout_RMSE_m=check_rms,holdout_spread_m=held_spread)
    for i in range(len(pa)):
        points.append(dict(source_east_m=float(east[i]),source_north_m=float(north[i]),
          destination_east_m=float(tr.c+pb[i,0]),destination_north_m=float(tr.f-pb[i,1]),
          residual_m=float(res[i]),role='fit' if fit[i] else 'holdout',inlier=bool(train[i] or check[i])))
    if train.sum()<12 or train.sum()/fit.sum()<.35 or train_rms>1 or spread<minimum_spread or np.linalg.norm(delta)>10:
        return dict(rec,status='rejected_fit_support'),points
    if check.sum()<6 or check.sum()/(~fit).sum()<.5 or check_rms>1 or held_spread<150:
        return dict(rec,status='rejected_heldout_support'),points
    aligned=cv2.warpAffine(a,np.float32([[1,0,dx],[0,1,dy]]),(a.shape[1],a.shape[0]),flags=cv2.INTER_LINEAR)
    shiftedmask=cv2.warpAffine(mask.astype('uint8'),np.float32([[1,0,dx],[0,1,dy]]),(a.shape[1],a.shape[0]),flags=cv2.INTER_NEAREST).astype(bool)
    test=binary_erosion(mask&shiftedmask,iterations=2)
    before=float(np.corrcoef(a[test],b[test])[0,1]);after=float(np.corrcoef(aligned[test],b[test])[0,1])
    rec.update(correlation_before=before,correlation_after=after,correlation_gain=after-before,correlation_pixels=int(test.sum()))
    return dict(rec,status='accepted' if np.isfinite(after-before) and after-before>=.015 else 'rejected_correlation_gain'),points
