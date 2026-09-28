"""Idempotency regression: hp.normalize v1.0.0 (frozen copy) vs the current module.

For the three fuzz sets of test_normalize.py (same generators and seeds) it counts
inputs whose output is not a fixed point (normalize(out) != out, or out not NFC)
under each version, and classifies the v1.0.0 failures by the rules that fire on
the second pass. It also counts inputs whose output changed between versions and
checks that each of them contains ZWJ, ZWNJ or the ALLAH ligature (the only
behaviour touched by 1.0.1).

On the released permissive corpus it compares both versions record by record
(text and change report) and hashes the concatenated outputs.

Output: regression/idempotency_regression.json.  Deterministic.
Run:    PYTHONIOENCODING=utf-8 python regression/regression_idempotency.py
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
import unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # test_normalize.py
sys.path.insert(0, r'F:\Hindko\_pipeline')
import test_normalize as T                          # noqa: E402
from hp import normalize as NEW                     # noqa: E402

spec = importlib.util.spec_from_file_location('normalize_v1_0_0', os.path.join(HERE, 'normalize_v1_0_0.py'))
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)
assert OLD.NORMALIZATION_VERSION == '1.0.0', OLD.NORMALIZATION_VERSION

ZWJ, ZWNJ, ALLAH_LIG = chr(0x200D), chr(0x200C), chr(0xFDF2)


def hexs(s):
    return ' '.join('%04X' % ord(c) for c in s)


def fixed_point_failures(mod, samples):
    bad, classes, examples = 0, Counter(), []
    for s in samples:
        a = mod.normalize(s)
        b, r = mod.normalize_with_report(a)
        if a != b or r or not unicodedata.is_normalized('NFC', a):
            bad += 1
            classes['+'.join(sorted(r)) or '(non-NFC)'] += 1
            if len(examples) < 5:
                examples.append({'input': hexs(s), 'first': hexs(a), 'second': hexs(b)})
    return bad, classes, examples


def main():
    t0 = time.time()
    res = {'old_version': OLD.NORMALIZATION_VERSION, 'new_version': NEW.NORMALIZATION_VERSION,
           'unicode_version': NEW.UNICODE_VERSION, 'fuzz': {}, 'corpus': {}}
    for label, gen in (('basic', T.fuzz_strings), ('focused', T.fuzz_focused), ('broad', T.fuzz_broad)):
        samples = gen()
        ob, ocls, oex = fixed_point_failures(OLD, samples)
        nb, ncls, nex = fixed_point_failures(NEW, samples)
        changed = 0
        changed_without_trigger = 0
        for s in samples:
            if OLD.normalize(s) != NEW.normalize(s):
                changed += 1
                if ZWJ not in s and ZWNJ not in s and ALLAH_LIG not in s:
                    changed_without_trigger += 1
        res['fuzz'][label] = {'strings': len(samples),
                              'v1_0_0_failures': ob, 'v1_0_0_failure_classes': dict(ocls.most_common()),
                              'v1_0_0_examples': oex,
                              'current_failures': nb, 'current_failure_classes': dict(ncls), 'current_examples': nex,
                              'outputs_differing_between_versions': changed,
                              'differing_without_ZWJ_ZWNJ_or_ALLAH_ligature': changed_without_trigger}
        print('%-8s %7d strings: v1.0.0 failures %5d %s | current failures %d | outputs differ %d '
              '(without ZWJ/ZWNJ/U+FDF2: %d)' % (label, len(samples), ob, dict(ocls.most_common()), nb, changed,
                                                 changed_without_trigger), flush=True)

    n = diff_text = diff_rep = 0
    h_old, h_new = hashlib.sha256(), hashlib.sha256()
    with open(r'F:\Hindko\hindko_dataset_permissive.jsonl', encoding='utf-8') as f:
        for line in f:
            t = json.loads(line)['text']
            ot, orep = OLD.normalize_with_report(t)
            nt, nrep = NEW.normalize_with_report(t)
            n += 1
            diff_text += ot != nt
            diff_rep += orep != nrep
            h_old.update(ot.encode('utf-8') + b'\x00')
            h_new.update(nt.encode('utf-8') + b'\x00')
    res['corpus'] = {'records': n, 'records_text_differs': diff_text, 'records_report_differs': diff_rep,
                     'sha256_outputs_v1_0_0': h_old.hexdigest(), 'sha256_outputs_current': h_new.hexdigest()}
    print('corpus: %d records, text differs %d, report differs %d, sha256 equal %s'
          % (n, diff_text, diff_rep, h_old.hexdigest() == h_new.hexdigest()))
    res['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, 'idempotency_regression.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
