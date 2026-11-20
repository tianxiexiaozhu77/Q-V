"""Audit saved MaCo outputs and source; no image/model inference or GPU usage."""
import collections,hashlib,json,math,pathlib
import numpy as np
BASE=pathlib.Path(__file__).resolve().parents[2]
MODEL=BASE/'MaCo'
RUN=BASE/'runs/matched_20261022_183356_2424149/MaCo_full_1'
DATA=BASE/'evaluation/matched_phase6_20261022'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
d=json.loads((DATA/'node21.json').read_text());rows=d['rows']
saved=json.loads((RUN/'node21/spatial_predictions.json').read_text())
peaks=collections.Counter(tuple(x[0]['peak_xy']) for x in saved)
boxes=np.asarray([b for row in rows for b in row['boxes']])
hits=[];plateau_intersections=[]
for row,items in zip(rows,saved):
 item=items[0];x,y=item['peak_xy']
 assert 0<=x<512 and 0<=y<512 and item['valid_fraction']==1
 hit=any(b[0]<=x<=b[2] and b[1]<=y<=b[3] for b in row['boxes'])
 assert int(hit)==item['hit'];hits.append(int(hit))
 # Exact nearest geometry: 14->224->512; 224 is divisible by 14.
 # This finds the FIRST maximum tile only. Other equal maximum tiles, if any,
 # were not retained for every image and are not assumed absent.
 gx=x*14//512;gy=y*14//512
 x2=math.ceil((gx+1)*512/14)-1;y2=math.ceil((gy+1)*512/14)-1
 plateau_intersections.append(int(any(max(x,math.ceil(b[0]))<=min(x2,math.floor(b[2])) and max(y,math.ceil(b[1]))<=min(y2,math.floor(b[3])) for b in row['boxes'])))
sample_maps=[]
with np.load(RUN/'node21/map_examples.npz') as maps:
 for key in maps.files:
  i=int(key.split('_')[-1]);m=maps[key][0];maximum=m==m.max();ys,xs=np.where(maximum)
  exact_any_max_hit=any(np.any(maximum[max(0,math.ceil(b[1])):min(512,math.floor(b[3])+1),max(0,math.ceil(b[0])):min(512,math.floor(b[2])+1)]) for b in rows[i]['boxes'])
  sample_maps.append({'image_index':i,'image':rows[i]['image'],'shape':list(m.shape),'finite':bool(np.isfinite(m).all()),'min':float(m.min()),'max':float(m.max()),'std':float(m.std()),'first_peak_xy':saved[i][0]['peak_xy'],'maximum_pixel_count':int(maximum.sum()),'maxima_extent_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],'boxes':rows[i]['boxes'],'shared_first_max_hit':hits[i],'any_max_pixel_hit_in_same_original_bbox_protocol':bool(exact_any_max_hit)})
weights=json.loads((MODEL/'local_setup/grounding_checkpoint_weight_audit.json').read_text())
validation=json.loads((MODEL/'local_setup/validation.json').read_text())
protocol=[
 {'stage':'image','author':'PIL direct bicubic resize 224x224; grayscale3; tensor; normalize mean .4978/std .2449; no center crop for grounding. The original discarded img.convert return is retained in behavior.','adapter':'same'},
 {'stage':'text','author':'Single supplied positive phrase, lowercase sentence processing, leading [CLS], released tokenizer; no negative prompt or positive-minus-negative map.','adapter':'There is Nodule. is the fixed AlphaRAD shared query; exact single-sentence author token processing. Classification four-negative prompts are NOT used for grounding.'},
 {'stage':'image_features','author':'Drop CLS from 197 tokens; retain 196 patches in row-major order; apply learned img_mlp to each 768-d patch.','adapter':'same'},
 {'stage':'spatial_weights','author':'w=softmax(checkpoint pos_weight_img.weight / .02); multiply each patch by its own weight.','adapter':'same; values read from official checkpoint, not regenerated center weights'},
 {'stage':'map_score','author':'Raw projected patch dot positive text vector. No L2 normalization, softmax over patches, extra temperature, positive-negative subtraction, or heatmap minmax at generation.','adapter':'same'},
 {'stage':'smoothing','author':'Reshape row-major 14x14; scipy.ndimage.gaussian_filter sigma=(1.5,1.5), order=0.','adapter':'same'},
 {'stage':'geometry','author':'Nearest 14x14 to224x224 through convert_similarity_to_image_size(width=height=resize=crop=224).','adapter':'same native224 map, then nearest to actual original H/W. Node21 is512 square, no crop, no unobserved margin, full finite coverage.'},
 {'stage':'checkpoint_pooler','author':'Source inherits HF384->384 pooler, but released checkpoint dense is[768,384] and bias[768], and image/text alignment expects768.','adapter':'Linear dimensions restored to checkpoint384->768; original first-token selection and Tanh retained; all trained weights loaded. Independent full author model was strict-loaded with the same dimensional compatibility fix.'},
 {'stage':'author_map_metric_normalization','author':'Pipeline.test normalizes raw heatmap monotonically to[-1,1] before PG.','adapter':'Shared PG uses raw map. Monotone minmax does not change extrema/argmax; first batch maps are nonconstant and finite.'},
 {'stage':'pointing_game_ties','author':'eval/metric.py compute_pg treats ALL equal maximum pixels as peaks; any overlap with GT mask is a hit.','adapter':'Shared AlphaRAD metric uses the FIRST maximum pixel (np.argmax), then any inclusive bbox. This is an intentional shared-metric protocol, NOT author-native PG. Nearest interpolation makes plateaus, so the distinction matters.'}
]
sources=[MODEL/'Phrase Grounding.py',MODEL/'eval/common.py',MODEL/'eval/metric.py',MODEL/'eval/utils.py',MODEL/'model_MaCo.py',MODEL/'bert/bert.py',MODEL/'local_setup/adapter.py',MODEL/'local_setup/verify_grounding.py',RUN/'node21/result.json',RUN/'node21/spatial_predictions.json',RUN/'node21/map_examples.npz',DATA/'node21.json',MODEL/'local_setup/validation.json']
report={'status':'complete_read_only_audit','scope':'Saved predictions/maps, official source and checkpoint metadata; no new inference, no GPU, no algorithm or result changes','official_repo':'https://github.com/SZUHvern/MaCo','official_commit':'bda781f3064273218d02b8d5a9103a0ab2bf98da','checkpoint_sha256':json.loads((MODEL/'local_setup/manifest.json').read_text())['checkpoint_sha256'],'cpu_author_map_parity':validation['grounding_native_parity'],'protocol_comparison':protocol,'node21':{'rows':len(rows),'class_names':d['class_names'],'shared_pg_hits':sum(hits),'shared_pg':sum(hits)/len(hits),'peak_distribution_xy':[{'xy':list(k),'count':v,'fraction':v/len(rows)} for k,v in peaks.most_common()],'bbox_count':len(boxes),'bbox_min_xyxy':boxes.min(0).tolist(),'bbox_max_xyxy':boxes.max(0).tolist(),'bbox_median_xyxy':np.median(boxes,0).tolist(),'boxes_containing_most_common_peak':sum(any(b[0]<=256<=b[2] and b[1]<=293<=b[3] for b in row['boxes']) for row in rows),'all_peak_coordinates_valid':True,'all_coverage_fractions_one':True,'first_maximum_tile_bbox_overlap_cases':sum(plateau_intersections),'first_maximum_tile_note':'Diagnostic only: first known maximum14x14 tile nearest-expanded to512; count is a lower bound for ANY-maximum overlap in the same original-bbox protocol. Not a recomputation of author-native PG (author mask quantization/protocol differs), not a replacement headline score.','saved_first_batch_maps':sample_maps},'checkpoint_weight_audit':weights,'conclusion':{'implementation_bug_found':False,'precise_scope':'No discrepancy found in continuous map generation, checkpoint weights, row-major geometry, coordinate range, saved-peak hit recomputation, or class/row identity. CPU independent-author-model parity is exact on the recorded validation example, not exhaustive proof over1129 images.','observed_failure':'Predictions concentrate around central image locations while annotated Node21 nodules do not contain those first maximum points; all1129 first-peak hits are zero.','protocol_limit':'The shared metric chooses one peak; author-native PG accepts every tied maximum. Therefore0.00 is the result of this checkpoint/query/native-heatmap transfer UNDER THE SHARED FIRST-PEAK PROTOCOL, not a claim that the MaCo paper or every author-native Node21 PG score is zero.','causality_limit':'This read-only audit does not isolate causal contributions of text prompt, coarse14x14 grid, Gaussian smoothing, spatial prior, or domain shift. Spatial-weight max/min ratio is only~1.377, so weights alone must not be blamed for the concentration.','task_transfer_limit':'No dedicated Node21 loader is present in the official release. Author ChestXDet10 example uses Findings suggesting [disease]. and224 masks; this campaign explicitly uses shared There is [disease]. and original-coordinate boxes. No prompt/interpolation/normalization/tie-policy selection was performed after viewing low scores.'},'source_sha256':{str(p):sha(p) for p in sources}}
(BASE/'MACO_GROUNDING_PROTOCOL_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
top=peaks.most_common(3)
txt=f'''MaCo Node21 定位协议核查（只读，2026-10-22）

结论：未找到连续热图生成或坐标/计分实现错误。当前 Node21 PG=0/1129，
是固定正提示、原生 MaCo 热图迁移到 Node21、再用共享 AlphaRAD 单峰 PG 的结果。
不能把它表述为 MaCo 原论文或作者原生 PG 协议在 Node21 上一定是0。
未重新推理、未使用GPU、未改变模型/提示/热图/阈值/现有结果。

发现一个必须注明的评估协议差别：
作者 eval/metric.py::compute_pg 接受所有并列最大像素，任一与GT mask相交即命中；
本次共享PG使用 np.argmax 返回的第一个最大像素，判断其是否落入任意GT框。
MaCo 作者原生 nearest 插值会生成最大值平台（Node21约36x36像素），因此这两种
PG不能混称。现有0.00遵守所有baseline共享的单峰协议，不应按低分改用更有利规则。

已存结果的直接证据：
- 1129行、1类 Nodule、512x512坐标，完整热图覆盖率均为1。
- 按已存peak与原始bbox重新计算1129次命中，全部与结果一致：0。
- 峰值最常见三处：{top}。
- {top[0][1]}/1129 图的首个峰值在(256,293)，所有GT框均不包含该点。
- bbox均在512图像坐标内：最小xyxy={boxes.min(0).tolist()}，
  最大xyxy={boxes.max(0).tolist()}；不是将原分辨率框误用到512图上的情况。
- 首批8张已存热图均finite、非恒定；逐张范围/标准差/最大平台像素数/框与峰坐标
  已写 JSON。首批原图峰值以胸部中央位置为主。
- 已知首个最大14x14网格单元扩展平台与GT框有交集的病例数为
  {sum(plateau_intersections)}（仅只读平台诊断下界，不是作者原生PG重跑或替代分数）。

逐操作与作者代码对照：
1. 图像：原生直接bicubic缩放224x224，灰度3通道，(.4978,.2449)归一化；无中心裁剪。
2. 文本：单个正短语，There is Nodule.，原作者单句小写处理、[CLS]和tokenizer。
   定位没有负文本；分类中1正+4负的softmax并没有混入热图。
3. 特征：去掉CLS，196个patch经img_mlp；乘softmax(pos_weight_img/.02)。
4. 相似度：未L2归一化的patch与文本原始点积，14x14行优先reshape，Gaussian sigma1.5。
5. 几何：作者nearest14→224；本次再nearest224→原图。Node21原图512方形，完整覆盖。
6. 作者Pipeline在指标前把map线性minmax到[-1,1]；这是单调变换，不改变非恒定map
   的峰位置。这里共享PG直接用rawmap，差别的关键在并列峰规则。
7. 权重：官方checkpoint epoch50、SHA256={report['checkpoint_sha256']}。
   pooler源码HF默认384→384与官方权重[768,384]/[768]不符，适配仅恢复384→768，
   保留原first-token+Tanh并加载全部参数。独立完整作者模型strict加载、原生Engine
   与适配器连续热图的CPU核验误差0；核验样本为ChestXDet10/Atelectasis，不声称
   对1129张做了全量独立原生重跑。
8. pos_weight为checkpoint实际学习值，softmax权重范围
   [{weights['softmax_weight_min']:.8f},{weights['softmax_weight_max']:.8f}]，
   最大/最小约{weights['softmax_weight_max_min_ratio']:.3f}；不能仅凭中心峰分布断言
   是空间权重造成的，未做去权重/改提示/改插值等择优消融。

迁移边界：官方loader没有Node21专用分支；原作者ChestXDet10示例提示是
Findings suggesting [disease].，并在224掩码坐标评估。本轮按AlphaRAD统一提示
There is [disease].、原图框与共享单峰PG评估。中心集中、粗网格及平台规则共同
构成需要报告的局限；本次未用测试结果挑选更好的提示或评分方式。

完整证据、源码SHA、首批数值及权重网格见 MACO_GROUNDING_PROTOCOL_AUDIT.json。
官方来源：https://github.com/SZUHvern/MaCo
commit：bda781f3064273218d02b8d5a9103a0ab2bf98da
'''
(BASE/'MACO_GROUNDING_PROTOCOL_AUDIT.txt').write_text(txt)
print(json.dumps({'audit_json':str(BASE/'MACO_GROUNDING_PROTOCOL_AUDIT.json'),'audit_txt':str(BASE/'MACO_GROUNDING_PROTOCOL_AUDIT.txt'),'hits':sum(hits),'rows':len(rows),'first_maximum_tile_overlap_cases':sum(plateau_intersections),'sample_ranges':[(x['min'],x['max']) for x in sample_maps]},ensure_ascii=False))
