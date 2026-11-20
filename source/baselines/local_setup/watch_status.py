#!/usr/bin/env python3
"""Publish download readiness and full evaluation state without changing jobs."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def snapshot(run):
    state = json.loads((run / 'status.json').read_text())
    lines = ['公开预训练baseline下载、环境与固定测试集评估', '',
             '更新：'+datetime.now().isoformat(timespec='seconds'),
             '控制器状态：'+state['status'], '当前批次：'+str(run),
             '正式对照表：'+str(run / 'RESULTS.txt'),
             '运行说明：'+str(ROOT / 'RUN_LOCAL.txt'), '',
             '| 模型 | 完整权重/环境/CPU核验 | GPU评估状态 | 当前任务/进度 |',
             '|---|---|---|---|']
    for name, job in state['models'].items():
        ready = ROOT / name / 'local_setup/READY.json'
        ok = ready.is_file() and json.loads(ready.read_text()).get('ready') is True
        status = job['status']
        if status == 'waiting_for_verified_assets' and ok:
            status = '已就绪，等待空闲GPU'
        detail = ''
        output = job.get('full_output') or job.get('smoke_output')
        if output:
            progress = Path(output) / 'progress.json'
            if progress.is_file():
                v = json.loads(progress.read_text())
                detail = f"{v['dataset']} {v['images_completed']}/{v['images_total']}"
        lines.append(f"| {name} | {'已通过' if ok else '准备/核验中'} | {status} | {detail} |")
    lines += ['', '只使用物理GPU6、7，使用前检查实际占用；没有模型训练。',
              '11个数据集、13个任务/测试范围；SIIM分别报告1704与1694张。',
              '只有full目录的complete任务进入正式结果；smoke为接口检查。',
              '原始方法与查询修正方法引用既有同范围结果，后者使用带标签支持样本。',
              'RSNA/SIIM的跨模型比较对齐身份及标签，并不保证DICOM解码/PNG转换像素完全一致。',
              'QaTa采用共享原图热图后处理及探索性测试阈值搜索；不宣称与AlphaRAD token归一化顺序逐操作相同。']
    target = ROOT / 'STATUS.txt'
    temp = target.with_suffix('.tmp'); temp.write_text('\n'.join(lines)+'\n'); temp.replace(target)
    return state['status']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    while True:
        status = snapshot(args.run.resolve())
        if args.once or status in ('complete', 'failed', 'interrupted'):
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
