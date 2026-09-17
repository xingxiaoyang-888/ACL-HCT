"""Stdlib-only registration and release binding for the matched E2 control."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


PROTOCOL = 'E2-matched-encoder-control-v1'
CONFIG_SHA256 = 'e07cb6244dcf6a25b25f9a463e90e1f953c3245a1578674799ae4010c16f691d'
SOURCE_NAMES = ('__init__ backbone geometry aggregation protocols ranking train mechanisms '
                'backbone_frozen_inputs encoder_matched encoder_matched_registration '
                'encoder_matched_control encoder_matched_entry').split()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    root = Path(__file__).parent
    return {name + '.py': hashlib.sha256((root / (name + '.py')).read_text(encoding='utf-8').encode()).hexdigest()
            for name in SOURCE_NAMES}


def validate_config(config):
    if canonical(config) != CONFIG_SHA256 or config.get('protocol') != PROTOCOL:
        raise ValueError('exact frozen encoder-matched registration required')
    return CONFIG_SHA256


def source_identity(declared):
    if not isinstance(declared, str) or not re.fullmatch('[0-9a-f]{40}', declared):
        raise ValueError('full lowercase published source commit required')
    root = Path(__file__).resolve().parents[2]
    observed = None
    try:
        top = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        if Path(top).resolve() == root:
            observed = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    if observed is not None and observed != declared:
        raise ValueError('declared source differs from available repository')
    return {'source_commit': declared, 'source_commit_basis': 'caller_declared',
            'git': {'available': observed is not None, 'commit': observed}}


def verify_release(config, identity, approval, quality_path):
    """Fail before opening prepared labels; bind exact user/reviewed release."""
    validate_config(config)
    required = {'user_authorized': True, 'quality_review_passed': True,
                'entry_criteria_frozen': True, 'supervisor_released': True,
                'scope': PROTOCOL, 'config_sha256': CONFIG_SHA256,
                'source_commit': identity['source_commit']}
    if (any(approval.get(k) != v or (isinstance(v, bool) and approval.get(k) is not v)
            for k, v in required.items())
            or not isinstance(approval.get('user_message_reference'), str)
            or not approval['user_message_reference'].strip()):
        raise ValueError('user authorization and supervisor release for exact code/config required')
    if file_sha256(quality_path) != approval.get('quality_record_sha256'):
        raise ValueError('reviewed engineering quality raw SHA mismatch')
    quality = json.loads(Path(quality_path).read_text(encoding='utf-8'))
    hashes = source_hashes()
    if (quality.get('status') != 'passed' or quality.get('config_canonical_sha256') != CONFIG_SHA256
            or quality.get('source_lf_sha256') != hashes):
        raise ValueError('engineering quality must bind exact current source/config')
    frozen = config['original_source_lf_sha256']
    for name, expected in frozen.items():
        actual = hashlib.sha256((Path(__file__).parent / name).read_text(encoding='utf-8').encode()).hexdigest()
        if actual != expected:
            raise ValueError('original scientific dependency changed: ' + name)
    return {'approval_record_canonical_sha256': canonical(approval),
            'quality_record_raw_sha256': file_sha256(quality_path), 'source_lf_sha256': hashes}
