"""Attack suite corresponding to user's 20 x 3 table.
All attacks return uint8 BGR 512x512 images. Implementation-specific
conventions (JPEG2000 quality layers, crop/resize, noise) are documented.
"""
from __future__ import annotations
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageEnhance,ImageFilter

LEVELS=('Mild','Moderate','Strong')
ATTACKS={
 'Compression':{
  'JPEG quality':(90,70,50),
  'JPEG2000 quality-layer parameter':(3,7,10),
  'Bit depth':(7,6,5),
  'WebP lossy quality':(90,70,50)},
 'Noise':{
  'Gaussian variance [0,1]':(.001,.003,.005),
  'Salt and pepper amount':(.001,.005,.01),
  'Speckle variance':(.001,.005,.01),
  'Poisson peak':(255,128,64)},
 'Filtering':{
  'Median size':(3,5,7),
  'Average size':(3,5,7),
  'Gaussian blur radius':(.5,1.,1.5),
  'Sharpening factor':(1.25,1.5,2.)},
 'Geometry':{
  'Direct rotation deg':(5,10,15),
  'Resize factor':(.9,.75,.5),
  'Center crop keep fraction':(.95,.9,.8),
  'Translation pixels':((2,2),(4,4),(8,8))},
 'Photometric':{
  'Brightness factor':(.95,.9,.8),
  'Contrast factor':(.95,.9,.8),
  'Gamma':(1.05,1.2,1.5),
  'Saturation factor':(.8,.5,.2)},
}

def _clip(a):return np.uint8(np.clip(np.rint(a),0,255))
def _pil(bgr):return Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),'RGB')
def _cv(im):return cv2.cvtColor(np.asarray(im.convert('RGB')),cv2.COLOR_RGB2BGR)

def attack(im,attack_name,level,seed=2026):
    if isinstance(level,str):i=LEVELS.index(level)
    else:i=int(level)
    settings={k:v for group in ATTACKS.values() for k,v in group.items()}
    v=settings[attack_name][i]
    rng=np.random.default_rng(int(seed))
    h,w=im.shape[:2]
    if attack_name=='JPEG quality':
        ok,encoded=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,int(v)])
        if not ok:raise RuntimeError('JPEG encode failed')
        return cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    if attack_name=='JPEG2000 quality-layer parameter':
        b=BytesIO();_pil(im).save(b,format='JPEG2000',quality_mode='rates',quality_layers=[int(v)])
        b.seek(0);return _cv(Image.open(b))
    if attack_name=='Bit depth':
        shift=8-int(v)
        return ((im.astype(np.uint16)>>shift)<<shift).astype(np.uint8)
    if attack_name=='WebP lossy quality':
        buffer=BytesIO()
        _pil(im).save(buffer,format='WEBP',quality=int(v),lossless=False,method=6)
        buffer.seek(0)
        return _cv(Image.open(buffer))
    if attack_name=='Gaussian variance [0,1]':
        noise=rng.normal(0,255*np.sqrt(float(v)),im.shape)
        return _clip(im.astype(np.float64)+noise)
    if attack_name=='Salt and pepper amount':
        out=im.copy();m=rng.random((h,w));out[m<float(v)/2]=0;out[(m>=float(v)/2)&(m<float(v))]=255;return out
    if attack_name=='Speckle variance':
        return _clip(im.astype(float)*(1+rng.normal(0,np.sqrt(float(v)),im.shape)))
    if attack_name=='Poisson peak':
        peak=float(v)
        return _clip(rng.poisson(im.astype(float)*peak/255)*255/peak)
    if attack_name=='Median size':return cv2.medianBlur(im,int(v))
    if attack_name=='Average size':return cv2.blur(im,(int(v),int(v)))
    if attack_name=='Gaussian blur radius':
        return _cv(_pil(im).filter(ImageFilter.GaussianBlur(radius=float(v))))
    if attack_name=='Sharpening factor':
        return _cv(ImageEnhance.Sharpness(_pil(im)).enhance(float(v)))
    if attack_name=='Direct rotation deg':
        M=cv2.getRotationMatrix2D((w/2,h/2),float(v),1)
        return cv2.warpAffine(im,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    if attack_name=='Resize factor':
        small=cv2.resize(im,(round(w*float(v)),round(h*float(v))),interpolation=cv2.INTER_AREA)
        return cv2.resize(small,(w,h),interpolation=cv2.INTER_CUBIC)
    if attack_name=='Center crop keep fraction':
        # Crop and resize back to original dimensions; this is a zoom crop.
        hh=round(h*float(v));ww=round(w*float(v));crop=im[(h-hh)//2:(h-hh)//2+hh,(w-ww)//2:(w-ww)//2+ww]
        return cv2.resize(crop,(w,h),interpolation=cv2.INTER_CUBIC)
    if attack_name=='Translation pixels':
        dx,dy=v;M=np.float32([[1,0,dx],[0,1,dy]])
        return cv2.warpAffine(im,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    if attack_name=='Brightness factor':return _clip(im.astype(float)*float(v))
    if attack_name=='Contrast factor':return _clip((im.astype(float)-127.5)*float(v)+127.5)
    if attack_name=='Gamma':return _clip(255*(im.astype(float)/255)**float(v))
    if attack_name=='Saturation factor':
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);a=hsv.astype(np.float64)
        a[:,:,1]=np.clip(a[:,:,1]*float(v),0,255)
        return cv2.cvtColor(_clip(a),cv2.COLOR_HSV2BGR)
    raise KeyError(attack_name)


def all_cases():
    for cat,attacks in ATTACKS.items():
        for name,settings in attacks.items():
            for i,level in enumerate(LEVELS):
                yield cat,name,level,settings[i]
