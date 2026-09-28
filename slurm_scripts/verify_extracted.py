"""Check extracted CL-VISTA annotations and referenced files, without GPU/network."""
import json
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('data', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--exclude-task', action='append', default=[],
                    help='Report missing references to this task without failing validation')
parser.add_argument('--allow-unreferenced-empty', action='store_true',
                    help='Report unreferenced empty videos; referenced empty videos still fail')
args = parser.parse_args()
data, output = args.data, args.output
excluded = set(args.exclude_task)
tasks = ['Counting', 'GUI', 'Movie', 'Science', 'Sports', 'STAR', 'Traffic']
report = {}
failed = False
video_suffixes = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpeg', '.mpg'}


def video_paths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('video', 'video_path'):
                if isinstance(item, str):
                    yield item
                elif isinstance(item, list):
                    yield from (p for p in item if isinstance(p, str))
            elif isinstance(item, (dict, list)):
                yield from video_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from video_paths(item)


for task in tasks:
    files = [p for p in (data / task).rglob('*') if p.is_file()]
    videos = [p for p in files if p.suffix.lower() in video_suffixes]
    annotation_dirs = [data / task / d for d in ('json_row', 'train_json_row', 'test_json_row')]
    annotation_dirs.append(data / 'json_row_unified' / task)
    annotations = sorted({p for folder in annotation_dirs for p in folder.rglob('*.json')})
    result = {'video_files': len(videos), 'video_bytes': sum(p.stat().st_size for p in videos),
              'zero_byte_videos': [str(p.relative_to(data)) for p in videos if p.stat().st_size == 0],
              'annotations': {}}
    failed |= not videos or not annotations
    failed |= bool(result['zero_byte_videos']) and not args.allow_unreferenced_empty
    all_refs, all_missing = set(), set()
    for path in annotations:
        try:
            records = json.loads(path.read_text())
            refs = set(video_paths(records))
            missing = sorted(p for p in refs if not (data / p).is_file())
            missing_in_scope = [p for p in missing if Path(p).parts[0] not in excluded]
            empty = sorted(p for p in refs if (data / p).is_file() and (data / p).stat().st_size == 0)
            all_refs.update(refs)
            all_missing.update(missing)
            result['annotations'][str(path.relative_to(data))] = {
                'records': len(records), 'unique_video_refs': len(refs), 'missing': missing,
                'missing_in_scope': missing_in_scope, 'empty_references': empty}
            failed |= bool(missing_in_scope) or bool(empty)
        except (ValueError, OSError, TypeError) as error:
            result['annotations'][str(path.relative_to(data))] = {'error': str(error)}
            failed = True
    result.update(unique_video_refs=len(all_refs), missing_video_refs=sorted(all_missing))
    missing_in_scope = sorted(p for p in all_missing if Path(p).parts[0] not in excluded)
    result.update(missing_in_scope=missing_in_scope, excluded_tasks=sorted(excluded))
    result['empty_references'] = sorted({p for a in result['annotations'].values()
                                         for p in a.get('empty_references', [])})
    report[task] = result
    print(f'{task}: videos={len(videos)}, annotation_files={len(annotations)}, '
          f'unique_references={len(all_refs)}, missing_in_scope={len(missing_in_scope)}, '
          f'missing_excluded={len(all_missing) - len(missing_in_scope)}, '
          f'empty_references={len(result["empty_references"])}, '
          f'zero_byte_video_files={len(result["zero_byte_videos"])}', flush=True)
output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(f'Report: {output}', flush=True)
if failed:
    raise SystemExit('Verification found missing/empty files or invalid annotations; see report')
print('ALL_IN_SCOPE_REFERENCED_VIDEO_FILES_EXIST', flush=True)
