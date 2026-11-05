"""Bounded local TrackNet retries. Predictions are verified, never fabricated."""
import numpy as np


def plan_recovery(rows, width, height, fps, budget=4):
    plans=[]; i=0
    while i<len(rows):
        if int(rows[i]['Visibility']): i+=1; continue
        start=i
        while i<len(rows) and not int(rows[i]['Visibility']): i+=1
        end=i
        # No cold-start guessing or retries across long occlusions/cuts.
        if start<3 or end-start>min(12,max(2,round(.4*fps))): continue
        anchors=[j for j in range(start-3,start) if int(rows[j]['Visibility'])]
        if len(anchors)<2: continue
        xy=np.array([[float(rows[j]['X']),float(rows[j]['Y'])] for j in anchors])
        if np.linalg.norm(xy[-1]-xy[-2])>40: continue
        center=xy[-1]
        if end<len(rows):
            right=np.array([float(rows[end]['X']),float(rows[end]['Y'])])
            if np.linalg.norm(right-center)<120: center=(center+right)/2
        w=min(width,384); h=min(height,216)
        x=int(np.clip(center[0]-w/2,0,width-w)); y=int(np.clip(center[1]-h/2,0,height-h))
        plans.append(dict(start=start,end=end,roi=[x,y,w,h],
                          selected=list(range(max(0,start-6),min(len(rows),end+4)))))
    # Longer gaps cannot be repaired by ordinary short-gap interpolation.
    return sorted(plans,key=lambda p:(-(p['end']-p['start']),p['start']))[:budget]


def merge_recovery(rows, local, plan, gate, fps):
    observed={int(r['Frame']):r for r in local if int(r['Visibility']) and gate.allows(float(r['X']),float(r['Y']))}
    def xy(r): return np.array([float(r['X']),float(r['Y'])])
    agreements=[i for i,r in observed.items() if int(rows[i]['Visibility']) and np.linalg.norm(xy(r)-xy(rows[i]))<=12]
    if len(agreements)<2: return [],'insufficient_anchor_agreement'
    accepted=[]
    limit=40*30/fps
    for i in range(plan['start'],plan['end']):
        if i not in observed or int(rows[i]['Visibility']): continue
        # At least two neighbouring real detections, including an anchored chain.
        neighbors=[j for j in range(max(0,i-2),min(len(rows),i+3)) if j!=i and j in observed
                   and np.linalg.norm(xy(observed[i])-xy(observed[j]))<=limit*abs(i-j)]
        if len(neighbors)<2: continue
        accepted.append(i)
    # Require connection to a previously observed track, not an isolated new object.
    connected=set(agreements)
    for _ in range(len(accepted)+1):
        connected.update(i for i in accepted if any(abs(i-j)<=2 and np.linalg.norm(xy(observed[i])-xy(observed[j]))<=limit*abs(i-j) for j in list(connected)))
    accepted=[i for i in accepted if i in connected]
    for i in accepted: rows[i]=dict(observed[i])
    return accepted,'recovered' if accepted else 'no_verified_detection'
