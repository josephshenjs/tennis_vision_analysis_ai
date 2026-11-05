"""Broadcast-style landing minimap and speed card without racket/pose graphics."""
from functools import lru_cache
import os
from pathlib import Path
import cv2
import numpy as np
from .compat.court_geometry import KEYPOINTS_M, LINES, LENGTH_M, WIDTH_M


def visible_shots(report,frame):
    return [e for e in report.get('events',[]) if e.get('landing_available_frame',e['available_frame'])<=frame and 'landing_court_m' in e]


@lru_cache(maxsize=128)
def speed_text(width,height,value):
    """Render TrueType at twice output resolution, then downsample for clear text.
    Use Windows' installed Segoe UI without bundling proprietary font files.
    """
    scale=width/162; factor=2
    try:
        from PIL import Image,ImageDraw,ImageFont
        fonts=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
        def font(size,bold=False):
            names=[str(fonts/('segoeuib.ttf' if bold else 'segoeui.ttf')),
                   'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf']
            for name in names:
                try: return ImageFont.truetype(name,max(1,round(size*scale*factor)))
                except OSError: pass
            raise OSError('No TrueType font available')
        layer=Image.new('RGB',(width*factor,height*factor)); draw=ImageDraw.Draw(layer)
        def xy(x,y): return (round(x*scale*factor),round(y*scale*factor))
        draw.text(xy(12,240),'Speed',font=font(15,True),fill=(44,188,113))
        centre=xy(128,0)[0]; number=font(22,True); unit=font(11,True)
        number_width=draw.textlength(value,font=number)
        draw.text((round(centre-number_width/2),xy(0,226)[1]),value,font=number,fill=(255,255,255))
        unit_width=draw.textlength('km/h',font=unit)
        draw.text((round(centre-unit_width/2),xy(0,253)[1]),'km/h',font=unit,fill=(236,237,240))
        return cv2.resize(np.asarray(layer)[:,:,::-1],(width,height),interpolation=cv2.INTER_AREA)
    except (ImportError,OSError):
        layer=np.zeros((height,width,3),np.uint8)
        cv2.putText(layer,'Speed',(round(12*scale),round(238*scale)),0,.46*scale,(220,220,220),max(1,round(scale)),cv2.LINE_AA)
        cv2.putText(layer,value+' km/h',(round(12*scale),round(262*scale)),0,.65*scale,(255,255,255),max(1,round(scale)),cv2.LINE_AA)
        return layer


def draw_shot_overlay(image,report,frame):
    height,width=image.shape[:2]; scale=min(width/1143,height/546)
    panel_w=max(1,round(162*scale)); panel_h=max(1,round(278*scale))
    # Supersample court graphics instead of enlarging already-rasterized small text.
    supersample=3
    panel=np.zeros((278*supersample,162*supersample,3),np.uint8)
    def point(x,y): return (round((18+(x+1.5)/(WIDTH_M+3)*126)*supersample),round((16+(y+2)/(LENGTH_M+4)*190)*supersample))
    for a,b in LINES: cv2.line(panel,point(*KEYPOINTS_M[a]),point(*KEYPOINTS_M[b]),(160,160,160),supersample,cv2.LINE_AA)
    cv2.line(panel,point(0,LENGTH_M/2),point(WIDTH_M,LENGTH_M/2),(210,210,210),supersample,cv2.LINE_AA)
    shots=visible_shots(report,frame)
    for event in shots[-8:]:
        x,y=event['landing_court_m']
        if not -1.5<=x<=WIDTH_M+1.5 or not -2<=y<=LENGTH_M+2: continue
        color=(0,190,90) if y<LENGTH_M/2 else (0,180,255)
        cv2.circle(panel,point(x,y),4*supersample,color,-1,cv2.LINE_AA)
        if event is shots[-1]: cv2.circle(panel,point(x,y),7*supersample,(245,245,245),supersample,cv2.LINE_AA)
    latest=next((e for e in reversed(shots) if e.get('best_prediction') and e['available_frame']<=frame),None)
    cv2.line(panel,(96*supersample,228*supersample),(96*supersample,272*supersample),(90,90,90),supersample,cv2.LINE_AA)
    resized=cv2.resize(panel,(panel_w,panel_h),interpolation=cv2.INTER_AREA)
    value='--' if latest is None else f'{latest["best_prediction"]["speed_kmh"]:.0f}'
    resized=cv2.max(resized,speed_text(panel_w,panel_h,value))
    margin=max(1,round(6*scale)); x=width-panel_w-margin; y=margin
    region=image[y:y+panel_h,x:x+panel_w]
    mask=np.zeros((panel_h,panel_w),np.uint8)
    def card(top,bottom,radius):
        left=0; right=panel_w-1; top=round(top*scale); bottom=min(panel_h-1,round(bottom*scale))
        radius=max(1,round(radius*scale))
        cv2.rectangle(mask,(left+radius,top),(right-radius,bottom),255,-1)
        cv2.rectangle(mask,(left,top+radius),(right,bottom-radius),255,-1)
        for cx,cy in [(left+radius,top+radius),(right-radius,top+radius),(left+radius,bottom-radius),(right-radius,bottom-radius)]:
            cv2.circle(mask,(cx,cy),radius,255,-1,cv2.LINE_AA)
    card(0,214,12); card(222,277,10)
    opacity=(mask.astype(float)/255*.68)[:,:,None]
    background=(region*(1-opacity)).astype(np.uint8)
    ink=np.any(resized>0,axis=2); background[ink]=resized[ink]
    region[:]=background
    return image
