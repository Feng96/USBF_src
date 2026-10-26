"""Plot the actual local product on its actual source RGB image."""
from pathlib import Path
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from .common import read
from usbf.data.sensors import load_sensor

def plot(data_dir,product,output):
    data_dir=Path(data_dir);example=data_dir/'example';selection=read(example/'selection.json')
    records=read(example/'sensors.json')['records'];r=next(r for r in records if r['name']==selection['anchor'])
    r=dict(r,path=str(example/r['path']));s=load_sensor(r,verify=True)
    frame=gpd.read_parquet(product).set_index('key')
    g=frame.geometry.to_crs(s['crs'])
    original=frame.gba_geometry.to_crs(s['crs'])
    height,width=s['signed'].shape;bounds=rasterio.transform.array_bounds(height,width,s['transform'])
    fig,axs=plt.subplots(1,len(selection['objects']),figsize=(12,4),squeeze=False)
    for ax,row in zip(axs.flat,selection['objects']):
        key=row['key'];a,b=original.loc[key],g.loc[key]
        ax.imshow(np.moveaxis(s['rgb'],0,-1),extent=[bounds[0],bounds[2],bounds[1],bounds[3]])
        gpd.GeoSeries([a],crs=s['crs']).boundary.plot(ax=ax,color='#ffcc33',linewidth=2,label='GBA')
        gpd.GeoSeries([b],crs=s['crs']).boundary.plot(ax=ax,color='#33ddff',linewidth=1.3,label='USBF')
        l,u,r,bounds_top=a.union(b).buffer(8).bounds;ax.set_xlim(l,r);ax.set_ylim(u,bounds_top)
        ax.set_title(row['role']+'\n'+key,fontsize=8);ax.set_aspect('equal');ax.tick_params(labelsize=6)
    fig.legend(handles=[plt.Line2D([0],[0],color='#ffcc33',label='Original GBA'),plt.Line2D([0],[0],color='#33ddff',label='Generated USBF')],loc='lower center',ncol=2)
    fig.tight_layout(rect=[0,.09,1,1]);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=250,bbox_inches='tight');plt.close(fig)
    return dict(output=str(output),actual_source_RGB=True,generated_geometry=True)
