# Asset attribution and terms

Software, trained parameters, source polygons, reference labels and imagery are separate materials. The author-code and checkpoint license choice is pending author confirmation. No license for author code is assigned by this candidate document. Vendored geometric packages retain their original MIT notices under `licenses/`.

| Material | Attribution and upstream terms |
|---|---|
| NAIP imagery and prepared RGB arrays | USDA National Agriculture Imagery Program; [official catalog](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery) |
| DSM/DTM and prepared signed height | USGS 3D Elevation Program and recorded contributing projects; [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) |
| SpaceNet-2 reference polygons and image context | SpaceNet Partners; [SpaceNet-2 license and citation instructions](https://spacenet.ai/spacenet-buildings-dataset-v2/), CC BY-SA 4.0; [license](https://creativecommons.org/licenses/by-sa/4.0/) |
| Original GBA geometry and derived outlines | Xiao Xiang Zhu, Sining Chen, Fahong Zhang, Yilei Shi and Yuanyuan Wang, GlobalBuildingAtlas; [component-specific source notice](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/main/README.md) |
| Archived OSM confirmation geometry | © OpenStreetMap contributors; [copyright and ODbL](https://www.openstreetmap.org/copyright) |
| Inherited Microsoft provenance | Microsoft building-footprint source identified by the original GBA source field; not a new independently downloaded truth layer |

SpaceNet citation: Van Etten, A., Lindenbaum, D., and Bacastow, T. M. (2018), *SpaceNet: A Remote Sensing Dataset and Challenge Series*, arXiv:1807.01232. Cropping, horizontal warping, reference translation and metric derivation in this package are documented transformations; original reference polygons remain available in the frozen snapshot.

GBA's official notice distinguishes ODbL polygons (`GBA.ODbLPolygon`) from its other CC BY-NC 4.0 products. The supplied original context inventory was inspected: 5,425 Microsoft-origin and 2,972 OSM-origin polygons; the primary fixed sample contains 1,436 and 583, respectively. These are the ODbL-origin polygon categories identified in the provider’s official notice. This example contains no newly generated GBA-source polygon category and does not distribute GBA height-map or LoD1 assets. GBA-derived polygon snapshots retain ODbL attribution and conditions, independently of the SpaceNet layers’ CC BY-SA terms. This component audit does not relicense unrelated national GBA products.

The release data index records object names, byte sizes and SHA-256 values. Prepared snapshots are the fixed reproduction inputs. Original acquisition object IDs and unsigned locations are recorded separately; a later download from an upstream live service is a new acquisition unless its content is verified against the recorded snapshot. No access token or time-limited signed URL belongs in the public asset registry.
