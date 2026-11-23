"""Render saved KAIST model outputs and controller resolutions; no new inference."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start',type=int,default=800)
    parser.add_argument('--frames',type=int,default=200)
    args=parser.parse_args()
    result=json.loads(args.run.read_text())
    manifest=json.loads(args.manifest.read_text())
    rows=result['rows']; indices=[r['frame_index'] for r in rows]
    selected=[r for r in rows if args.start<=r['frame_index']<args.start+args.frames]
    if not selected:raise ValueError('Empty segment')
    if args.output.exists():raise ValueError('Preserve existing animation')
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans'
    fonts={size:ImageFont.truetype(font+suffix+'.ttf',size)
           for size,suffix in [(12,''),(14,''),(16,''),(18,'-Bold'),(23,'-Bold'),(29,'-Bold')]}
    bg='#0b1220'; ink='#edf4ff'; muted='#a7b8d1'; accent='#54e2ce'
    names=['Bus','Car','Lamp','Motorcycle','Person','Truck']
    colors=['#ffae52','#63beff','#ffe87d','#ec98f3','#54e2ce','#c4a1ff']
    width,height=1100,752
    actions=[480,544,576,640]
    trace_left,trace_right,trace_top,trace_bottom=72,1068,637,706
    def tx(index):return trace_left+(index-args.start)/(args.frames-1)*(trace_right-trace_left)
    def ty(res):return trace_bottom-(res-480)/160*(trace_bottom-trace_top)
    animation=[]
    preview=None
    selected_indices=[]
    for source_index in range(args.start,args.start+args.frames,2):
        position=bisect_right(indices,source_index)-1
        if position<0:continue
        row=rows[position]; frame=manifest['frames'][row['frame_index']]
        selected_indices.append(row['frame_index'])
        im=Image.new('RGB',(width,height),bg); draw=ImageDraw.Draw(im)
        draw.text((22,17),'ADAPTIVE RESOLUTION  /  RGB + THERMAL',font=fonts[23],fill=ink)
        draw.text((23,53),'YOLO26n  ·  frozen sequence controller  ·  KAIST '+manifest['sequence'],font=fonts[14],fill=muted)
        draw.rounded_rectangle((830,15,1078,73),radius=12,fill='#173a3b')
        draw.text((847,29),f"{row['resolution']} px  INPUT",font=fonts[23],fill=accent)
        detections=[d for d in row['detections'] if d[4]>=.25 and d[2]>d[0] and d[3]>d[1]]
        for kind,x,label in [('rgb',22,'VISIBLE / RGB'),('thermal',562,'THERMAL / LWIR')]:
            path=Path(frame[kind])
            if not path.is_absolute():path=args.manifest.parent/path
            with Image.open(path) as src:
                image=src.convert('RGB'); native_w,native_h=image.size
                image=image.resize((516,413),Image.LANCZOS)
            boxes=ImageDraw.Draw(image)
            for x1,y1,x2,y2,confidence,cls in sorted(detections,key=lambda d:d[5]==4):
                cls=int(cls); color=colors[cls]
                box=(max(0,x1*516/native_w),max(0,y1*413/native_h),min(515,x2*516/native_w),min(412,y2*413/native_h))
                boxes.rectangle(box,outline=color,width=2 if cls!=4 else 3)
                text=f'{names[cls]} {confidence:.2f}'
                tw=boxes.textlength(text,font=fonts[12]); left=min(box[0],max(0,515-tw-8)); top=max(0,box[1]-19)
                boxes.rectangle((left,top,left+tw+7,top+18),fill=bg)
                boxes.text((left+3,top+1),text,font=fonts[12],fill=color)
            draw.text((x,90),label,font=fonts[14],fill=muted)
            im.paste(image,(x,114))
        draw=ImageDraw.Draw(im)
        current_time=frame['timestamp_s']
        switches=sum(a['resolution']!=b['resolution'] for a,b in zip(selected,selected[1:]) if b['frame_index']<=row['frame_index'])
        metrics=[('SOURCE TIME',f'{current_time:.2f} s'),('DETECTIONS',str(len(detections))),
                 ('SERVICE TIME',f"{row['service_ms']:.1f} ms"),('RESOLUTION CHANGES',str(switches))]
        for x,(label,value) in zip([22,292,562,832],metrics):
            draw.rounded_rectangle((x,543,x+246,611),radius=9,fill='#142136')
            draw.text((x+12,550),label,font=fonts[12],fill=muted)
            draw.text((x+12,570),value,font=fonts[23],fill=ink)
        for action in actions:
            y=ty(action);draw.line((trace_left,y,trace_right,y),fill='#23354b')
            draw.text((24,y-7),str(action),font=fonts[12],fill=muted)
        points=[(tx(r['frame_index']),ty(r['resolution'])) for r in selected if r['frame_index']<=row['frame_index']]
        for a,b in zip(points,points[1:]):draw.line((a,(b[0],a[1]),b),fill=accent,width=3)
        if points:
            x,y=points[-1];draw.ellipse((x-4,y-4,x+4,y+4),fill=ink)
            draw.line((x,trace_top-4,x,trace_bottom+4),fill='#65879a')
        draw.text((22,727),'Recorded processed-frame replay · boxes ≥ 0.25 · native image display · selected for visible switching',font=fonts[12],fill=muted)
        if len(animation)==40:preview=im.copy()
        animation.append(im.quantize(colors=192,method=Image.MEDIANCUT))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    animation[0].save(args.output,save_all=True,append_images=animation[1:],duration=100,loop=0,optimize=False,disposal=2)
    if preview is not None:preview.save(args.output.with_suffix('.png'))
    switches=sum(a['resolution']!=b['resolution'] for a,b in zip(selected,selected[1:]))
    metadata=dict(run=str(args.run),manifest=str(args.manifest),sequence=manifest['sequence'],
        source_frame_range=[args.start,args.start+args.frames-1],rendered_frames=len(animation),
        duration_seconds=len(animation)/10,resolutions=sorted({r['resolution'] for r in selected}),
        recorded_resolution_changes=switches,displayed_confidence=.25,
        protocol='Same-frame boxes from saved processed outputs; source-index sampling at 10 fps. Native image display, not simulated lower-resolution imagery. Not a live capture or causal display-age demonstration.',
        selection='Illustrative '+str(manifest['frames'][0].get('environment',''))+' segment selected after evaluation for visible detections and resolution changes; not representative performance evidence.',
        source='KAIST Multispectral Pedestrian Detection Benchmark, Hwang et al., CVPR 2015',
        license='Dataset-derived imagery: CC BY-NC-SA 4.0; https://soonminhwang.github.io/rgbt-ped-detection/')
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),bytes=args.output.stat().st_size,**metadata),indent=2))


if __name__=='__main__':main()
